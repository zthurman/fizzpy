def answer(s):
    s = "Abc'S"
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o,'
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    upperalphabet = [x.upper() for x in alphabet]
    reversedalphabet = alphabet[::-1]
    symbol = "~`!@#$%^&*()_-+={}[]:>;',</?*-+"
    reversed_s = ""
    for i in s:
        case = i.islower()
        if case:
            if i in alphabet:
                index = alphabet.index(i)
                newletter = reversedalphabet[index]
                reversed_s += newletter
        elif not case:
            if i in upperalphabet:
                reversed_s += i
            elif i in symbol:
                reversed_s += i
    return reversed_s

if __name__ == '__main__':
    string = "Abc'S"
    print(answer(string))
