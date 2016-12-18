def answer(s):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o,'
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    reversedalphabet = alphabet[::-1]
    upperalphabet = [x.upper() for x in reversedalphabet]
    symbol = "~`!@#$%^&*()_-+={}[]:>;',</?*-+"
    space = " "
    reversed_s = ""
    for i in s:
        case = i.islower()
        if case:
            if i in reversedalphabet:
                index = reversedalphabet.index(i)
                newletter = alphabet[index]
                reversed_s += newletter
        elif not case:
            if i in upperalphabet:
                reversed_s += i
            elif i in symbol:
                reversed_s += i
            elif i == space:
                reversed_s += i
    return reversed_s

if __name__ == '__main__':
    string = "Zyx uid szh"
    print(answer(string))