import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText


def send_email(content):
    smtp_server = os.getenv("IONO_SMTP_SERVER")
    smtp_port = int(os.getenv("IONO_SMTP_PORT", "465"))
    sender = os.getenv("IONO_EMAIL_SENDER")
    password = os.getenv("IONO_EMAIL_PASSWORD")
    receiver = os.getenv("IONO_EMAIL_RECEIVER", sender or "")

    if not all([smtp_server, sender, password, receiver]):
        print(
            "email notification skipped: set IONO_SMTP_SERVER, "
            "IONO_EMAIL_SENDER, IONO_EMAIL_PASSWORD, and "
            "IONO_EMAIL_RECEIVER to enable it."
        )
        return

    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = Header("程序运行通知", "utf-8")

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
    except Exception as e:
        print(f"failed to send email: {e}")


if __name__ == "__main__":
    send_email("测试邮件")
