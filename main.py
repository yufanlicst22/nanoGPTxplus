"""
Main file for running the Language Modelling example with nanodoGPT
"""

import jax
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # yufanli@x-casing-447521-m1.iam.gserviceaccount.com
    # service-282424833597@cloud-tpu.iam.gserviceaccount.com
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("Hello from the print statement!")
    logger.info("Hello from the logger!")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


if __name__ == "__main__":
    main()