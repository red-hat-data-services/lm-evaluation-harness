import sys
import os
import logging
import subprocess
from pathlib import Path
from olot.oci_artifact import create_simple_oci_artifact
from olot.backend.skopeo import is_skopeo, skopeo_push, skopeo_inspect
from olot.oci.oci_utils import get_descriptor_from_manifest

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  {i}: {arg}")

    oci_ref = os.environ["OCI_REGISTRY"] + "/" + os.environ["OCI_REPOSITORY"] + ":" + os.environ["OCI_TAG"]
    logger.info(f"oci_ref: {oci_ref}")

    if not is_skopeo():
        logger.error("Missing requirement: skopeo must be installed.")
        exit -1
    try:
        result = subprocess.run(["skopeo", "--version"], capture_output=True, text=True, check=True)
        logger.info(f"skopeo version: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Failed to check skopeo version: {e}")
        exit -1

    subject = None
    if os.environ.get("OCI_SUBJECT"):
        if not os.environ["OCI_SUBJECT"].startswith(os.environ["OCI_REGISTRY"]) or os.environ["OCI_REPOSITORY"] not in os.environ["OCI_SUBJECT"]:
            logger.error("OCI Subject/referrer must be in the same registry and repository")
            exit -1
        if "@" not in os.environ["OCI_SUBJECT"]:
            logger.warning("Should prefer to use a digest based reference, instead of Tag reference")
        manifest = skopeo_inspect("docker://"+os.environ["OCI_SUBJECT"], skopeo_params(""))
        subject = get_descriptor_from_manifest(manifest)

    # assumption is `output` directory sibling to this script containing the output traces (and results)
    this_script_path = Path(__file__).resolve()
    output_path = this_script_path.parent / ".." / "output"
    logger.info(f"Output path: {output_path}")
    if output_path.exists():
        subdirs = [d for d in output_path.iterdir() if is_valid_subdir(d)]
        if len(subdirs) == 1:
            output_path = subdirs[0]
            logger.info(f"Output path updated to single subdirectory: {output_path}")
    list_files(output_path)

    oci_layout_path = Path("/tmp/oci-layout")
    oci_layout_path.mkdir(parents=True, exist_ok=True)
    create_simple_oci_artifact(output_path, oci_layout_path, subject=subject)
    logger.info(f"/tmp/oci-layout: {oci_layout_path}")
    list_files(oci_layout_path)

    logger.info("Transfering oci-layout of OCI Artifact (traces and results) to destination...")
    skopeo_push(oci_layout_path, oci_ref, skopeo_params("dest-"))


def is_valid_subdir(path: Path) -> bool:
    """Return True if this is a subdirectory we want to consider containing results and traces"""
    return path.is_dir() and path.name != "lost+found"


def skopeo_params(infix: str = ""):
    """
    Common function to compute the Skopeo parameters
    
    :param infix: either leave empty or pass "dest-" to infix resulting skopeo parameters
    :type infix: str
    """
    skopeo_params = []
    docker_config_path = Path("/tmp/.docker/config.json")
    if docker_config_path.exists():
        skopeo_params.append("--"+infix+"authfile")
        skopeo_params.append("/tmp/.docker/config.json")
    else:
        skopeo_params.append("--"+infix+"username")
        skopeo_params.append(os.environ["OCI_USERNAME"])
        skopeo_params.append("--"+infix+"password")
        skopeo_params.append(os.environ["OCI_PASSWORD"])
    if os.environ.get("OCI_VERIFY_SSL"):
        skopeo_params.append("--"+infix+"tls-verify=false")
    return skopeo_params


def list_files(path):
    logger.debug(f"\nRecursive listing of {path}:")
    for root, _, files in os.walk(path):
        level = root.count(os.sep)
        indent = "." * level
        logger.debug(f"{indent}{os.path.basename(root)}")
        subindent = "." * (level + 1)
        for f in files:
            logger.debug(f"{subindent}{f}")


if __name__ == "__main__":
    main()
