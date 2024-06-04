export PYTHONPATH=$PYTHONPATH:/workspace/code

# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2024_audioldm_encodec/audioldm_encodec.yaml

# Train the VAE
# python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml