# nanoGPT

Yufan's research project repo. 

# Installation
1. Set up the gcloud CLI to use TPU VM follwoing the guide: https://cloud.google.com/sdk/docs/install-sdk
2. Request TPU instance following the guide https://cloud.google.com/tpu/docs/run-calculation-jax
   1. First run to request a TPU VM
      ```
      gcloud compute tpus tpu-vm create yufanfish \
      --zone=us-central1-a \
      --accelerator-type=v3-8 \
      --version=tpu-ubuntu2204-base
      ```
   2. Connection to VM
      ```
      gcloud compute tpus tpu-vm ssh yufanfish --zone=us-central1-a
      ```
3. Install Remote-SSH extension on VSCode
4. Run
   ```
   gcloud compute tpus tpu-vm ssh yufanfish \                       
    --zone=us-central1-a \
    --dry-run
   ```
   which should produce output
   ```
   /usr/bin/ssh -t -i /Users/yufanli/.ssh/google_compute_engine -o CheckHostIP=no -o HashKnownHosts=no -o HostKeyAlias=tpu.7519457668826798549-0-fl77Co -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o
   UserKnownHostsFile=/Users/yufanli/.ssh/google_compute_known_hosts yufanli@35.184.77.221
   ```
   Read the IP address: yufanli@35.184.77.221 and update config file in User/yufanli/.ssh/config to
   ```
   Host my-tpu
   HostName 35.184.77.221
   User yufanli
   IdentityFile ~/.ssh/google_compute_engine
   CheckHostIP no
   StrictHostKeyChecking no
   ```
5. Now find Remote Explorer side tab and connect to my-tpu instance
6. The above will open a new VSCode window within which git clone the desired repo
      
   
