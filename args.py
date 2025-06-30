import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')

    parser.add_argument('--step', type=float, default=1.0)  
    parser.add_argument('--step2d', type=float, default=1.0)
    parser.add_argument('--step2d_no', type=float, default=10000)
    parser.add_argument('--num_point', type=int, default=100)
    parser.add_argument('--x_range', type=int, default=70)
    parser.add_argument('--y_range', type=int, default=14)   
    parser.add_argument('--z_range', type=int, default=80)
    parser.add_argument('--r_ori', type=int, default=0.5)  
    parser.add_argument('--ratioE', type=int, default=0.065) 
    parser.add_argument('--gamma', type=int, default=3.0)
    parser.add_argument('--ratioQ', type=int, default=0.5)

    args = parser.parse_args()
    return args