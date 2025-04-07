import pandas as pd
import wandb
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import time


def extract_wandb_run_info(run):
    return {
        "path": "/".join([str(p) for p in run.path]),
        "state": run.state,
        "run_obj": run,
    }


def download_wandb_runs(
    entity: str,
    project: str,
    tags: Optional[List[str]] = None,
    max_workers: int = 10,
    cache_file: str = "wandb_runs_cache.pkl",
) -> pd.DataFrame:
    """
    Download runs data from Weights & Biases with parallelization and caching.

    Args:
        entity: The wandb entity (username or team name)
        project: The wandb project name
        tags: Optional list of tags to filter runs
        max_workers: Maximum number of parallel workers
        cache_file: Path to cache file to avoid re-downloading runs (default: "wandb_runs_cache.pkl")

    Returns:
        pandas.DataFrame: DataFrame containing the runs data
    """
    # Check if cache exists and load it
    cached_df = pd.DataFrame()
    try:
        cached_df = pd.read_pickle(cache_file)
        print(f"Loaded {len(cached_df)} runs from cache")
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load cache: {e}")

    api = wandb.Api()

    # Build the filter query
    filter_query = {}
    if tags:
        filter_query["tags"] = {"$in": tags}

    # Get runs from the specified project
    print("Fetching runs from W&B...")
    fetch_start = time.time()
    runs = api.runs(f"{entity}/{project}", filters=filter_query, per_page=250)
    print(f"Found {len(runs)} runs in {time.time() - fetch_start:.2f} seconds")

    # Extract basic run info first to use DataFrame for filtering
    print("Extracting basic run info...")
    run_info = []
    extract_start = time.time()
    for run in tqdm.tqdm(runs, desc="Extracting run info"):
        run_info.append(extract_wandb_run_info(run))

    print(
        f"Done extracting run info for {len(run_info)} runs in {time.time() - extract_start:.2f} seconds"
    )

    # Convert to DataFrame for faster filtering
    runs_df = pd.DataFrame(run_info)

    # Filter out runs that are already in the cache and have finished status
    if not cached_df.empty:
        # Create a mask for runs that need to be processed
        # Either not in cache or in cache but not finished
        cached_finished_runs = cached_df[cached_df["state"] == "finished"][
            "path"
        ].tolist()
        runs_df["needs_processing"] = ~runs_df["path"].isin(cached_finished_runs)
        new_runs = runs_df[runs_df["needs_processing"]]["run_obj"].tolist()
    else:
        new_runs = runs_df["run_obj"].tolist()

    print(f"Downloading {len(new_runs)} new runs")

    def process_run(run):
        """Process a single run and extract its data"""
        run_data = {
            "name": run.name,
            "path": "/".join([str(p) for p in run.path]),
            "state": run.state,
        }

        # Add config parameters
        for key, value in run.config.items():
            if key not in run_data:
                # Only store simple types to avoid recursion issues
                if isinstance(value, (int, float, str, bool)) or value is None:
                    run_data[f"config/{key}"] = value

        # Add summary metrics
        for key, value in run.summary.items():
            if key not in run_data:
                # Only store simple types to avoid recursion issues
                if isinstance(value, (int, float, str, bool)) or value is None:
                    run_data[key] = value

        return run_data

    # Extract run data in parallel for new runs
    new_runs_data = []
    if new_runs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_run = {executor.submit(process_run, run): run for run in new_runs}

            # Process results as they complete with a progress bar
            for future in tqdm.tqdm(
                as_completed(future_to_run), total=len(new_runs), desc="Processing runs"
            ):
                try:
                    run_data = future.result()
                    new_runs_data.append(run_data)
                except Exception as exc:
                    print(f"Run processing generated an exception: {exc}")

    # Convert new runs to DataFrame
    new_runs_df = pd.DataFrame(new_runs_data) if new_runs_data else pd.DataFrame()

    # Combine cached and new runs data
    if cached_df.empty:
        df = new_runs_df
    elif new_runs_df.empty:
        df = cached_df
    else:
        # Remove any runs from cache that are being updated
        if not new_runs_df.empty and "path" in new_runs_df.columns:
            cached_df = cached_df[~cached_df["path"].isin(new_runs_df["path"])]
        df = pd.concat([cached_df, new_runs_df], ignore_index=True)

    # Update cache if new runs were processed
    if not new_runs_df.empty:
        df.to_pickle(cache_file)
        print(f"Updated cache with {len(new_runs_df)} new runs")

    return df
