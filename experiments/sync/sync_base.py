from plato.clients import simple
from plato.datasources import base
from plato.servers import fedavg
from plato.trainers import basic
import numpy as np


def main():
    import torch
    random_seed = 69 # or any of your favorite number 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    client = simple.Client()
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()
