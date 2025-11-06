"""Standalone program for running ngen simulations with regionalized parameters.

This script calls MSWM to set up an NGEN simulation and runs the NGEN simulation
  for a specified VPU using regionalized parameters. It takes various command-line arguments
  to specify the run configuration and paths to necessary files.

Sometimes the NGEN simulation may fail due to issues in the routing module. In such cases,
this script will attempt to run `nwm_routing` as a fallback to generate the routing output.

See run_ngen_vpu.sh for an example of how to run this script.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from mswm.build_inputs import RealizationBuilder

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path, log_level: int = logging.INFO):
    """Set up logging configuration."""
    # log file path
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up root logger so that each module can log to the same file
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers (avoid duplicates)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (use 'w' to overwrite each run)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")


def expand_and_verify_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Expand user and resolve/validate paths in the given argparse Namespace."""
    for key, value in vars(args).items():
        if "dir" in key or "file" in key:
            p = Path(value).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"{p} does not exist")
            setattr(args, key, p)
    return args


def log_run_info(args: argparse.Namespace):
    """Log the run information."""
    logger.info("====== Settings for NGEN Regionalization Run ======")
    logger.info(f"VPU:            {args.vpu}")
    logger.info(f"Run name:       {args.run_name}")
    logger.info(f"Time range:     {args.start_time} → {args.end_time}")
    logger.info(f"Working dir:    {args.work_dir}")
    logger.info(f"Parameter file: {args.par_file}")
    logger.info(f"Pair file:      {args.pair_file}")
    logger.info(f"GPKG file:      {args.gpkg_file}")
    logger.info(f"MSWM template file:  {args.config_template_file}")
    logger.info(f"NGEN input dir:    {args.out_dir / '../Input'}")
    logger.info(f"NGEN output dir:     {args.out_dir}")
    logger.info("================================================")


def create_mswm_config(args: argparse.Namespace):
    """Create MSWM config file based on template."""
    with open(args.config_template_file, "r") as f:
        template_content = f.read()

    # format start and end times (as required by MSWM)
    start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")
    args.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    args.end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # Replace placeholders in the template
    config_content = template_content.format(
        vpu=args.vpu,
        run_name=args.run_name,
        start_time=args.start_time,
        end_time=args.end_time,
        par_file=args.par_file,
        pair_file=args.pair_file,
        gpkg_file=args.gpkg_file,
        work_dir=args.work_dir,
        nprocs=args.nprocs,
    )

    # Write the new config file
    config_path = Path(args.out_dir).parent / "mswm.config"
    with open(config_path, "w") as f:
        f.write(config_content)

    logger.info(f"Created MSWM config file at: {config_path}")
    return config_path


def verify_ngen_run_inputs(args: argparse.Namespace):
    """Verify that necessary input files for NGEN run exist."""
    input_dir = Path(args.out_dir / "../Input").resolve()
    ngen_exe = input_dir / "ngen"
    real_file = (
        Path(args.out_dir).parent / f"{args.vpu}_realization_config_bmi_region.json"
    )
    real_file = real_file.resolve()
    partition_file = input_dir / f"{args.vpu}_partition_config.json"
    hydrofab_file = input_dir / Path(args.gpkg_file).name

    # command line argument validation
    if not ngen_exe.is_file():
        raise FileNotFoundError(f"NGEN executable not found: {ngen_exe}")
    if not real_file.is_file():
        raise FileNotFoundError(f"Realization config file not found: {real_file}")
    if not partition_file.is_file() and args.nprocs > 1:
        raise FileNotFoundError(f"Partition file not found: {partition_file}")
    if not hydrofab_file.is_file():
        raise FileNotFoundError(f"Hydrofabric file not found: {hydrofab_file}")
    logger.info("All NGEN command-line arguments verified.")

    # TODO: verify the module BMI config file also exists

    return ngen_exe, real_file, partition_file, hydrofab_file


def run_mswm_with_unified_logging(config_path, log_level=logging.INFO):
    """Run MSWM RealizationBuilder to build realization and BMI config files."""
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers.copy()

    # Dedicated MSWM logger
    mswm_logger = logging.getLogger("mswm")
    mswm_logger.setLevel(log_level)
    mswm_logger.propagate = False
    mswm_logger.handlers.clear()
    for h in saved_handlers:
        mswm_logger.addHandler(h)

    # Disable propagation on all mswm sub-loggers to avoid double logging
    for name, logger_obj in logging.root.manager.loggerDict.items():
        if name.startswith("mswm") and isinstance(logger_obj, logging.Logger):
            logger_obj.handlers = mswm_logger.handlers
            logger_obj.propagate = False

    # Call MSWM
    rb = RealizationBuilder(config_path)
    rb.build_region_realization()

    # Restore root logger
    root_logger.handlers.clear()
    for h in saved_handlers:
        root_logger.addHandler(h)
    logging.getLogger("main").info("MSWM finished, logging restored.")


def build_ngen_command(args: argparse.Namespace) -> str:
    """Build the NGEN command string."""
    ngen_exe, real_file, partition_file, hydrofab_file = verify_ngen_run_inputs(args)

    if args.nprocs < 1:
        args.nprocs = 1
        logger.warning("Number of processors set to 1.")

    if args.nprocs == 1:
        logger.info("Running NGEN in serial mode.")
        cmd_str = f"""
        cd {args.out_dir}
        {ngen_exe} {hydrofab_file} all {hydrofab_file} all {real_file}
        """
    else:
        logger.info(f"Running NGEN in parallel mode with {args.nprocs} processors.")
        cmd_str = f"""
        cd {args.out_dir}
        mpirun --allow-run-as-root -n {args.nprocs} {ngen_exe} {hydrofab_file} all {hydrofab_file} all {real_file} {partition_file}
        """

    return cmd_str


def run_nwm_routing(args):
    """Run nwm_routing after NGEN failure (only if routing output does not exist)."""
    routing_logger = logging.getLogger("nwm_routing")
    routing_logger.propagate = False
    routing_logger.setLevel(args.log_level)

    routing_logger.handlers.clear()
    for h in logging.getLogger().handlers:
        routing_logger.addHandler(h)

    # routing config file
    routing_config = (
        Path(args.out_dir).parent / "Input" / f"{args.vpu}_troute_config_region.yaml"
    )
    if not routing_config.is_file():
        routing_logger.error(f"Routing config file not found: {routing_config}")
        raise FileNotFoundError(f"Routing config file not found: {routing_config}")

    # Parse and format start time (expects args.start_time like '2022-10-01T00:00:00')
    try:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
        start_time_str = start_time.strftime("%Y%m%d%H%M")
    except Exception:  # prevents double logging
        # Fallback: use raw string if parsing fails
        start_time_str = (
            str(args.start_time).replace(":", "").replace("-", "").replace("T", "")
        )

    # Expected routing output file
    routing_output = Path(args.out_dir) / f"troute_output_{start_time_str}.nc"

    # Check if output already exists
    if routing_output.exists():
        routing_logger.info(
            f"Routing output already exists: {routing_output}. Skipping nwm_routing run."
        )
        return

    routing_logger.info(
        f"Routing output not found ({routing_output}). Running fallback command to generate it."
    )

    # Keep track of LEVELPOOL warning lines to avoid duplicates
    warning_seen = set()
    warning_strs = ["WARNING: LEVELPOOL USING COLDSTART WATER ELEVATION"]

    # Build and run nwm_routing command
    cmd = ["python", "-m", "nwm_routing", "-f", "-V4", str(routing_config)]
    routing_logger.info(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=args.out_dir,  # Use cwd argument to change directory
    )

    for line in iter(process.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue

        if any(warning in line for warning in warning_strs):
            if line in warning_seen:
                continue
            warning_seen.add(line)

        routing_logger.info(line)

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        routing_logger.error(f"nwm_routing exited with code {return_code}")
        raise subprocess.CalledProcessError(return_code, cmd)
    else:
        routing_logger.info("nwm_routing completed successfully.")


def run_ngen_with_unified_logging(args):
    """Run NGEN as a subprocess and log output.

    If NGEN fails and t-route output file is not found, run nwm_routing as a fallback.

    """
    ngen_logger = logging.getLogger("ngen")
    ngen_logger.propagate = False
    ngen_logger.setLevel(args.log_level)

    # Attach root handlers so logs go to unified log file + console
    root_handlers = logging.getLogger().handlers
    ngen_logger.handlers.clear()
    for h in root_handlers:
        ngen_logger.addHandler(h)

    # Build command string for run NGEN
    cmd = build_ngen_command(args)

    # Keep track of warning lines to avoid duplicates
    warnings_seen = set()
    warning_strs = ["[BMI WARNING]"]

    process = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    try:
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue

            # Filter duplicate warnings
            if any(warning in line for warning in warning_strs):
                if line in warnings_seen:
                    continue
                warnings_seen.add(line)

            ngen_logger.info(line)

            # Detect abort messages
            if "Aborted" in line or "core dumped" in line:
                aborted = True
                ngen_logger.error("Detected NGEN abort. Terminating process...")
                process.terminate()
                break

    finally:
        process.stdout.close()
        return_code = process.wait()

    if return_code != 0 or aborted:
        ngen_logger.error(f"NGEN failed with return code {return_code}.")
        ngen_logger.info("Attempting fallback: running nwm_routing...")

        run_nwm_routing(args)
        return  # Exit after fallback

    ngen_logger.info("NGEN simulation finished successfully.")


def main_workflow(
    args: argparse.Namespace,
):
    """Run the main workflow to build MSWM config and execute MSWM and NGEN simulations."""
    # Log run information
    log_run_info(args)

    # create MSWM config file based on template
    input_path = create_mswm_config(args)

    # Run MSWM to build realization and module BMI config files
    run_mswm_with_unified_logging(input_path)

    # Run NGEN simulation
    run_ngen_with_unified_logging(args)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Standalone driver for running ngen simulation for a VPU with regionalized parameters."
    )
    parser.add_argument("--vpu", required=True, help="VPU identifier (e.g., vpu_09)")
    parser.add_argument("--run_name", required=True, help="Name of this run")
    parser.add_argument(
        "--work_dir", required=True, type=Path, help="Working directory"
    )
    parser.add_argument(
        "--start_time", required=True, help="Start time (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end_time", required=True, help="End time (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument("--par_file", required=True, help="Path to parameter file")
    parser.add_argument(
        "--pair_file", required=True, help="Path to donor/receiver pair file"
    )
    parser.add_argument(
        "--gpkg_file", required=True, help="Path to hydrofabric GPKG file"
    )
    parser.add_argument(
        "--config_template_file",
        required=True,
        help="Path to configuration template file",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=2,
        help="Number of processors for parallel NGEN run (default: 2)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (if not provided, defaults to region.log in output directory)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # get command line arguments
    args = get_args()

    # create output directory
    out_dir = args.work_dir / "regionalization" / args.run_name / args.vpu / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir

    # make sure log level is valid
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    args.log_level = numeric_level

    # set up logging
    setup_logging(
        log_file=args.log_file or out_dir.parent / "region.log",
        log_level=args.log_level,
    )

    # Expand paths
    args = expand_and_verify_paths(args)

    try:
        main_workflow(args)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)
