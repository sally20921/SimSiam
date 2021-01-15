import re

def main():
    dict1 = {'a': 'test00', 'b': 'test01'}
    
    dict2 = {'b': 'test02', 'a': 'test04'}

    dict3 =  {'c': 'test05','b': 'test02', 'a': 'test04'}
    print(sorted(dict3.values()))
    a,b,c = (v for k, v in sorted(dict3.items()))
    print(a,b,c)

if __name__ == "__main__":
    main()

