# nanoGPT

Yufan's research project repo. 

# Installation
1. Set up the gcloud CLI to use TPU VM follwoing the guide: https://cloud.google.com/sdk/docs/install-sdk
2. Request TPU instance following the guide https://cloud.google.com/tpu/docs/run-calculation-jax
   a. First run
   ```
   gcloud compute tpus tpu-vm create yufanfish \
   --zone=us-central1-a \
   --accelerator-type=v3-8 \
   --version=tpu-ubuntu2204-base
   ```
