#python -m anomapy.train.sassn -env BeamRider -latent_shape 64 -epochs 16 -batch_size 128 -colour True
#python -m anomapy.train.sassn -env Breakout -latent_shape 64 -epochs 16 -batch_size 128 -colour True 
#python -m anomapy.train.sassn -env Enduro -latent_shape 64 -epochs 16 -batch_size 128 -colour True
python -m anomapy.train.sassn -env Pong -latent_shape 256 -epochs 40 -batch_size 256 -colour True
#python -m anomapy.train.sassn -env Qbert -latent_shape 64 -epochs 16 -batch_size 128 -colour True  
#python -m anomapy.train.sassn -env Seaquest -latent_shape 64 -epochs 16 -batch_size 128 -colour True  
#python -m anomapy.train.sassn -env SpaceInvaders -latent_shape 64 -epochs 16 -batch_size 128 -colour True