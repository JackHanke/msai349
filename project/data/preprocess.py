
chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

# these two four loops describe data
for char in chars:
    for ind in range(1, 3000):
        path_str = f'./asl_alphabet_train/asl_alphabet_train/{char}/{char}{ind}.jpg'
        # TODO vectorize jpg files


if __name__ == '__main__':
    pass
