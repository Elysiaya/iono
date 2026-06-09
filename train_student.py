from iono.train_student import train_student
from scripts.send_email import send_email


if __name__ == "__main__":
    train_student()
    send_email("Student model FGL distillation training completed. Check checkpoints.")
