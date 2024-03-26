import argparse


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--name','-n',default='Deepika',type=str)
    args.add_argument('--age','-a',default=35.0,type=float)
    args_parse=args.parse_args()
    print(args_parse.name,args_parse.age)