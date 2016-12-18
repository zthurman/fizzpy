def answer(s):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    reversedalphabet = alphabet[::-1]
    upperalphabet = [x.upper() for x in reversedalphabet]
    symbol = "~`!@#$%^&*()_-+={}[]:>;',</?*-+"
    numbers = "0123456789"
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
            elif i == "\\":
                reversed_s += i
            elif i in numbers:
                reversed_s += i
    return reversed_s


def stringjumbler(s):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    upperalphabet = [x.upper() for x in alphabet]
    reversedalphabet = alphabet[::-1]
    symbol = "~`!@#$%^&*()_-+={}[]:>;',</?*-+"
    numbers = "0123456789"
    space = " "
    reversed_s = ""
    for i in s:
        case = i.islower()
        if case:
            if i in reversedalphabet:
                index = alphabet.index(i)
                newletter = reversedalphabet[index]
                reversed_s += newletter
        elif not case:
            if i in upperalphabet:
                reversed_s += i
            elif i in symbol:
                reversed_s += i
            elif i == space:
                reversed_s += i
            elif i == "\\":
                reversed_s += i
            elif i in numbers:
                reversed_s += i
    return reversed_s

if __name__ == '__main__':
    # string = "Yvzs! I xzm'g yvorvev Lzmxv olhg srh qly zg gsv xlolmb!!"
    # string = "wrw blf hvv ozhg mrtsg'h vkrhlwv?"
    string = "Eer\\mahgerdwat in dE Fa*&%[]i ng fre~1"
    # print(string)
    # reversedstring = stringjumbler(string)
    # print(reversedstring)
    print(answer(string))


