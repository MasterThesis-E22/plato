from plato.clients import simple
from plato.datasources import base
from plato.servers import fedavg
from plato.trainers import basic


def main():
    client = simple.Client()
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()
