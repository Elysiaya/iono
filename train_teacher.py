from iono.train_teacher import train_teacher
from scripts.send_email import send_email


if __name__ == "__main__":
    train_teacher()
    send_email("Teacher model FGL distillation training completed. Check checkpoints.")
