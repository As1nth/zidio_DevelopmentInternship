import smtplib

def send_alert(transaction_id, amount, to_email):
    sender = "sreebhanu1221@gmail.com"          # 👈 Replace with your Gmail
    app_password = "odkypilrnbugjxmp"       # 👈 Use Gmail App Password

    subject = "Fraud Alert Detected!"
    body = f"""
    Suspicious transaction detected!

    Transaction ID: {transaction_id}
    Amount:  {amount} INR


    Please review immediately.
    """

    message = f"Subject: {subject}\n\n{body}"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, to_email, message)
        server.quit()
        print("✅ Alert sent to", to_email)
    except Exception as e:
        print("❌ Failed to send alert:", e)
# Test the alert system
if __name__ == "__main__":
    send_alert(
        transaction_id=123456,
        amount=4500.75,
        to_email="sreebhanu1221@gmail.com"  # 👈 replace with your real Gmail
    )
