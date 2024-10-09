import argparse
from utils.trainer import train

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Add arguments
    parser.add_argument('--mode', type=str, choices=['img', 'pts', 'both'], default='img', 
                        help="Mode to run the model: 'img', 'pts', or 'both'")
    parser.add_argument('--data_dir', type=str, required=True, help="path to data dir")
    parser.add_argument('--log_name', type=str, required=True, help="Log file name")
    parser.add_argument('--resolution', type=int, choices=[20, 10], default=20, help="Resolution to use for the data")
    parser.add_argument('--num_epoch', type=int, required=True, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=48, help="Number of epochs to train the model")
    parser.add_argument('--use_mf', action='store_true', help="Use multi-fusion (set flag to enable)")
    parser.add_argument('--use_residual', action='store_true', help="Use residual connections (set flag to enable)")

    # Parse arguments
    args = parser.parse_args()
    
    # User specifies which datasets to use
    datasets_to_use = ['rmf_s2/spring/tiles_128','rmf_s2/summer/tiles_128','rmf_s2/fall/tiles_128','rmf_s2/winter/tiles_128']

    # Call the train function with parsed arguments
    train(
        data_dir=args.data_dir,
        datasets_to_use=datasets_to_use,
        resolution=args.resolution,
        log_name=args.log_name,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        mode=args.mode,
        use_mf=args.use_mf,
        use_residual=args.use_residual)
    

if __name__ == "__main__":
    main()
