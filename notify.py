from plyer import notification
import os
import sys

def send_notification(title, message):
    try:
        # Try Plyer notification
        notification.notify(
            title=title,
            message=message,
            timeout=10
        )
    except NotImplementedError:
        print("Plyer notification failed. Trying notify-send...")

        # Fallback to notify-send (Linux only)
        if os.system("which notify-send") == 0:
            os.system(f'notify-send "{title}" "{message}"')
        else:
            print("notify-send is not installed.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        send_notification(sys.argv[1], sys.argv[2])
    else:
        send_notification("Task Completed", "Your process has finished successfully!")

