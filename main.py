from data_exploit import get_data,type_dif
from sequence_model_prep import model_prep

def main():
    type_dif(get_data())
    model_prep()


# Chamada para a função principal.
if __name__ == '__main__':
    main()
