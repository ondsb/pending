"""
One-time conversion: CSV on S3 → Parquet on S3.

Usage:
    python convert_to_parquet.py          # full run
    python convert_to_parquet.py --test   # test with 10MB subset first
"""

import argparse
import logging
import os
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from pending_delay.config import settings
from pending_delay.schema import TICKET_SCHEMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

s3_cfg = settings.s3
SRC_BUCKET = s3_cfg.src_bucket
CSV_KEY = s3_cfg.csv_key
DST_BUCKET = s3_cfg.dst_bucket
PARQUET_KEY = s3_cfg.parquet_key
AGGREGATES = s3_cfg.aggregates

DATA_DIR = settings.data_dir
LOCAL_CSV = DATA_DIR / "tickets_raw.csv"
LOCAL_PARQUET = DATA_DIR / "tickets.parquet"

SCHEMA = TICKET_SCHEMA


def make_progress_cb(total_bytes, label):
    state = {"transferred": 0, "last_pct": -5}

    def cb(chunk_bytes):
        state["transferred"] += chunk_bytes
        pct = state["transferred"] / total_bytes * 100
        if pct - state["last_pct"] >= 5:
            state["last_pct"] = pct
            log.info(f"  {label}: {state['transferred']/1e9:.1f} / {total_bytes/1e9:.1f} GB ({pct:.0f}%)")

    return cb


def test_run(s3_src, s3_dst):
    """Quick end-to-end test with 10MB of data."""
    import io

    log.info("=== TEST RUN (10MB subset) ===")

    # 1. Read 10MB from source
    log.info("Reading 10MB from source CSV...")
    resp = s3_src.get_object(Bucket=SRC_BUCKET, Key=CSV_KEY, Range="bytes=0-10000000")
    raw = resp["Body"].read()
    raw = raw[: raw.rfind(b"\n") + 1]
    log.info(f"  got {len(raw)/1e6:.1f} MB")

    # 2. Convert to parquet in memory
    log.info("Converting to Parquet...")
    reader = pcsv.open_csv(io.BytesIO(raw), convert_options=pcsv.ConvertOptions(column_types=SCHEMA))
    test_parquet = DATA_DIR / "tickets_test.parquet"
    writer = None
    total = 0
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(str(test_parquet), batch.schema, compression="zstd")
        writer.write_batch(batch)
        total += batch.num_rows
    writer.close()
    size = test_parquet.stat().st_size
    log.info(f"  wrote {total:,} rows, {size/1e6:.1f} MB")

    # 3. Upload to destination
    test_key = "data/pending/tickets_test.parquet"
    log.info(f"Uploading to s3://{DST_BUCKET}/{test_key}...")
    s3_dst.upload_file(str(test_parquet), DST_BUCKET, test_key)
    log.info("  upload OK")

    # 4. Verify it's readable
    log.info("Verifying uploaded file...")
    head = s3_dst.head_object(Bucket=DST_BUCKET, Key=test_key)
    log.info(f"  verified: {head['ContentLength']/1e6:.1f} MB on S3")

    # Cleanup local test file
    test_parquet.unlink()
    log.info("=== TEST PASSED ===\n")


def download_aggregates(s3_src):
    """Download aggregate CSVs from S3 (cached locally). Returns dict of name → local path."""
    agg_local = {}
    for name, key in AGGREGATES.items():
        local_path = DATA_DIR / Path(key).name
        expected_size = s3_src.head_object(Bucket=SRC_BUCKET, Key=key)["ContentLength"]
        if local_path.exists() and local_path.stat().st_size == expected_size:
            log.info(f"  {name}: cached ({expected_size/1e6:.0f} MB)")
        else:
            log.info(f"  {name}: downloading {expected_size/1e6:.0f} MB...")
            t0 = time.time()
            s3_src.download_file(
                SRC_BUCKET, key, str(local_path),
                Config=TransferConfig(max_concurrency=10),
            )
            log.info(f"    done in {time.time()-t0:.0f}s")
        agg_local[name] = local_path
    return agg_local


def merge_aggregates(tickets_parquet: Path, output_parquet: Path, agg_local: dict):
    """DuckDB join: tickets parquet + aggregate CSVs → merged parquet."""
    import duckdb

    con = duckdb.connect()

    query = f"""
    COPY (
        WITH tickets AS (
            SELECT * FROM read_parquet('{tickets_parquet}')
        ),
        bs AS (
            SELECT * FROM read_csv_auto('{agg_local["bettor_stats"]}', header=true)
        ),
        rej AS (
            SELECT
                bettor_id, sport, market_type,
                SUM(rejected_stake) AS total_rejected_stake,
                SUM(rejected_pnl) AS total_rejected_pnl,
                COUNT(DISTINCT reject_reason) AS n_reject_reasons,
                AVG(avg_rejected_odds_after_10) AS avg_rejected_odds_after_10,
                AVG(avg_rejected_odds_after_30) AS avg_rejected_odds_after_30,
                AVG(avg_rejected_odds_after_90) AS avg_rejected_odds_after_90,
            FROM read_csv_auto('{agg_local["rejected"]}', header=true)
            GROUP BY bettor_id, sport, market_type
        ),
        rt AS (
            SELECT
                bettor_id,
                ARG_MAX(ots_risk_tier_id, volume) AS dominant_risk_tier,
                SUM(pnl) AS risk_tier_total_pnl,
                SUM(volume) AS risk_tier_total_volume,
                AVG(margin) AS risk_tier_avg_margin,
            FROM read_csv_auto('{agg_local["risk_tier"]}', header=true)
            GROUP BY bettor_id
        ),
        ss AS (
            SELECT * FROM read_csv_auto('{agg_local["stake_size"]}', header=true)
        )
        SELECT
            t.*,
            bs.pnl AS bs_pnl, bs.margin AS bs_margin, bs.stake AS bs_stake,
            bs.avg_odds_after_10 AS bs_avg_odds_after_10,
            bs.avg_odds_after_30 AS bs_avg_odds_after_30,
            bs.avg_odds_after_90 AS bs_avg_odds_after_90,
            bs.rejected_stake AS bs_rejected_stake, bs.rejected_pnl AS bs_rejected_pnl,
            rej.total_rejected_stake, rej.total_rejected_pnl, rej.n_reject_reasons,
            rej.avg_rejected_odds_after_10, rej.avg_rejected_odds_after_30, rej.avg_rejected_odds_after_90,
            rt.dominant_risk_tier, rt.risk_tier_total_pnl, rt.risk_tier_total_volume, rt.risk_tier_avg_margin,
            ss.mean_stake_size,
        FROM tickets t
        LEFT JOIN bs ON t.bettor_id = bs.bettor_id AND t.sport = bs.sport AND t.market_name = bs.market_type
        LEFT JOIN rej ON t.bettor_id = rej.bettor_id AND t.sport = rej.sport AND t.market_name = rej.market_type
        LEFT JOIN rt ON t.bettor_id = rt.bettor_id
        LEFT JOIN ss ON t.bettor_id = ss.bettor_id
    ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    log.info("Joining with aggregates...")
    t0 = time.time()
    con.sql(query)
    elapsed = time.time() - t0
    size = output_parquet.stat().st_size
    log.info(f"  done in {elapsed:.0f}s → {output_parquet} ({size/1e6:.1f} MB)")
    con.close()


def subset_run(s3_src, subset_mb=50):
    """Download a subset of ticket data via byte-range, join with aggregates on S3, write merged parquet.

    Uses boto3 byte-range to grab first N MB of the ticket CSV (avoids streaming 60GB).
    Converts to a temp local parquet, then DuckDB joins it with aggregates from S3.
    """
    import io

    subset_bytes = subset_mb * 1_000_000
    temp_tickets = DATA_DIR / "_tickets_subset.parquet"

    log.info(f"=== SUBSET RUN ({subset_mb}MB of tickets + aggregates) ===")

    # 1. Grab first N MB of ticket CSV via byte-range request
    log.info(f"Reading {subset_mb}MB from ticket CSV...")
    resp = s3_src.get_object(Bucket=SRC_BUCKET, Key=CSV_KEY, Range=f"bytes=0-{subset_bytes}")
    raw = resp["Body"].read()
    raw = raw[: raw.rfind(b"\n") + 1]
    log.info(f"  got {len(raw)/1e6:.1f} MB")

    # 2. Convert to temp local parquet
    log.info("Converting ticket subset to parquet...")
    reader = pcsv.open_csv(io.BytesIO(raw), convert_options=pcsv.ConvertOptions(column_types=SCHEMA))
    writer = None
    total = 0
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(str(temp_tickets), batch.schema, compression="zstd")
        writer.write_batch(batch)
        total += batch.num_rows
    writer.close()
    log.info(f"  {total:,} rows → {temp_tickets} ({temp_tickets.stat().st_size/1e6:.1f} MB)")

    # 3. Download aggregates and merge
    agg_local = download_aggregates(s3_src)
    merge_aggregates(temp_tickets, LOCAL_PARQUET, agg_local)

    # Cleanup temp ticket file (keep aggregates for reuse)
    temp_tickets.unlink()
    log.info("=== SUBSET READY — run: streamlit run app.py ===\n")


def full_run(s3_src, s3_dst):
    """Full conversion: download CSV → convert to Parquet → merge aggregates → upload → cleanup."""
    csv_size = s3_src.head_object(Bucket=SRC_BUCKET, Key=CSV_KEY)["ContentLength"]
    log.info(f"Source: s3://{SRC_BUCKET}/{CSV_KEY} ({csv_size / 1e9:.1f} GB)")
    log.info(f"Dest:   s3://{DST_BUCKET}/{PARQUET_KEY}")

    # Check disk space
    stat = os.statvfs(str(DATA_DIR))
    free_gb = (stat.f_bavail * stat.f_frsize) / 1e9
    needed_gb = csv_size / 1e9 + 15  # CSV + estimated parquet
    log.info(f"Disk: {free_gb:.0f} GB free, need ~{needed_gb:.0f} GB")
    if free_gb < csv_size / 1e9 + 5:
        raise RuntimeError(f"Not enough disk space: {free_gb:.0f} GB free, need at least {csv_size/1e9 + 5:.0f} GB")

    # Step 1: Download CSV
    if LOCAL_CSV.exists() and LOCAL_CSV.stat().st_size == csv_size:
        log.info(f"CSV already downloaded: {LOCAL_CSV}")
    else:
        if LOCAL_CSV.exists():
            LOCAL_CSV.unlink()
        log.info("Step 1/4: Downloading CSV from S3...")
        t0 = time.time()
        s3_src.download_file(
            SRC_BUCKET, CSV_KEY, str(LOCAL_CSV),
            Config=TransferConfig(max_concurrency=10),
            Callback=make_progress_cb(csv_size, "download"),
        )
        log.info(f"  done in {time.time() - t0:.0f}s")

    # Verify download
    actual_size = LOCAL_CSV.stat().st_size
    if actual_size != csv_size:
        raise RuntimeError(f"Download incomplete: {actual_size} != {csv_size}")

    # Step 2: Convert to Parquet
    log.info("Step 2/4: Converting CSV → Parquet...")
    t0 = time.time()

    reader = pcsv.open_csv(
        str(LOCAL_CSV),
        read_options=pcsv.ReadOptions(block_size=256 * 1024 * 1024),
        convert_options=pcsv.ConvertOptions(column_types=SCHEMA),
    )

    writer = None
    total_rows = 0
    batch_num = 0

    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(str(LOCAL_PARQUET), batch.schema, compression="zstd")
        writer.write_batch(batch)
        total_rows += batch.num_rows
        batch_num += 1
        elapsed = time.time() - t0
        log.info(f"  batch {batch_num} | {total_rows:>12,} rows | {elapsed:5.0f}s | {total_rows/elapsed:,.0f} rows/s")

    writer.close()
    parquet_size = LOCAL_PARQUET.stat().st_size
    log.info(
        f"  done: {total_rows:,} rows in {time.time()-t0:.0f}s | "
        f"{parquet_size/1e9:.2f} GB ({csv_size/parquet_size:.1f}x compression)"
    )

    # Step 3: Merge with aggregates
    log.info("Step 3/4: Downloading aggregates and merging...")
    agg_local = download_aggregates(s3_src)
    raw_parquet = DATA_DIR / "_tickets_raw.parquet"
    LOCAL_PARQUET.rename(raw_parquet)
    merge_aggregates(raw_parquet, LOCAL_PARQUET, agg_local)
    raw_parquet.unlink()
    parquet_size = LOCAL_PARQUET.stat().st_size
    log.info(f"  merged parquet: {parquet_size/1e9:.2f} GB")

    # Step 4: Upload Parquet
    # Refresh credentials before upload — SSO token may have expired during download+convert
    log.info("Refreshing destination credentials before upload...")
    s3_dst = boto3.Session(profile_name=s3_cfg.dst_profile).client("s3")

    log.info("Step 4/4: Uploading Parquet to S3...")
    t0 = time.time()
    s3_dst.upload_file(
        str(LOCAL_PARQUET), DST_BUCKET, PARQUET_KEY,
        Config=TransferConfig(multipart_chunksize=256 * 1024 * 1024, max_concurrency=10),
        Callback=make_progress_cb(parquet_size, "upload"),
    )
    log.info(f"  done in {time.time()-t0:.0f}s")

    # Verify upload
    head = s3_dst.head_object(Bucket=DST_BUCKET, Key=PARQUET_KEY)
    if head["ContentLength"] != parquet_size:
        raise RuntimeError(f"Upload verification failed: {head['ContentLength']} != {parquet_size}")
    log.info(f"Verified: s3://{DST_BUCKET}/{PARQUET_KEY} ({parquet_size/1e9:.2f} GB)")

    # Only delete CSV after upload is verified
    LOCAL_CSV.unlink()
    log.info(f"  deleted {LOCAL_CSV.name} to free disk space")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test with 10MB subset first")
    parser.add_argument("--subset", type=int, nargs="?", const=50, metavar="MB",
                        help="Build merged parquet locally from first N MB of ticket CSV (default 50)")
    parser.add_argument("--merge", action="store_true",
                        help="Just merge existing tickets.parquet with aggregates (skip CSV download/convert)")
    args = parser.parse_args()

    # Ensure data dir exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        if not LOCAL_PARQUET.exists():
            raise FileNotFoundError(f"No parquet to merge: {LOCAL_PARQUET}")
        s3_src = boto3.Session(profile_name=s3_cfg.src_profile).client("s3")
        agg_local = download_aggregates(s3_src)
        raw_parquet = DATA_DIR / "_tickets_raw.parquet"
        LOCAL_PARQUET.rename(raw_parquet)
        merge_aggregates(raw_parquet, LOCAL_PARQUET, agg_local)
        raw_parquet.unlink()
        log.info(f"Done: {LOCAL_PARQUET} ({LOCAL_PARQUET.stat().st_size/1e6:.1f} MB)")
    elif args.subset is not None:
        s3_src = boto3.Session(profile_name=s3_cfg.src_profile).client("s3")
        subset_run(s3_src, subset_mb=args.subset)
    elif args.test:
        s3_src = boto3.Session(profile_name=s3_cfg.src_profile).client("s3")
        s3_dst = boto3.Session(profile_name=s3_cfg.dst_profile).client("s3")
        test_run(s3_src, s3_dst)
    else:
        s3_src = boto3.Session(profile_name=s3_cfg.src_profile).client("s3")
        s3_dst = boto3.Session(profile_name=s3_cfg.dst_profile).client("s3")
        # Always run test first
        test_run(s3_src, s3_dst)
        full_run(s3_src, s3_dst)


if __name__ == "__main__":
    main()
