from src.tain_svr import Train_svr
from src.prepare_data import PrepareData

def menu():
    print("Welcome to Face Detection Using SVR and HOG")
    print("Please select from the following options:")
    print("1) Prepare data and save it into pickle file")
    print("2) Train model and evaluate")
    print("0) Exit")


if __name__ == '__main__':
    while True:
        menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            # Prepare data: Create and process data, saving batches into binary files
            try:
                preparer = PrepareData()
                preparer.process()
                print("Data preparation completed.")
            except Exception as e:
                print("An error occurred during data preparation:", e)

        elif choice == "2":
            # Train model and then evaluate it
            try:
                trainer = Train_svr()  # This class should inherit from TrainBase and override new_model
                trainer.new_model()
                trainer.read_data()
                trainer.split_data(percentage_of_train_data=0.8)
                trainer.train()
                trainer.evaluate()  # Evaluate the trained model on test data
            except Exception as e:
                print("An error occurred during training or evaluation:", e)

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please try again.")
