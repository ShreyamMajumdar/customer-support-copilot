import pandas as pd
import os

messages = [

    "I was charged twice", "I need a refund", "Wrong amount on invoice",
    "Please reverse my payment", "I was overcharged", "Refund not received",
    "Cancel my subscription", "Incorrect billing", "Duplicate charge",
    "My bill is too high",

    "App keeps crashing", "Cannot login", "Website not loading",
    "Password reset not working", "Getting error on screen",
    "App is very slow", "Cannot upload file", "Page is blank after login",
    "Notifications not working", "Search feature is broken",

    "Order not arrived", "Where is my package", "Delivery is delayed",
    "Wrong item delivered", "Package shows delivered but not received",
    "I want to change delivery address", "Package was damaged",
    "Order stuck in transit", "Missing items in order", "How long is shipping",

    "Cannot access my account", "How to delete account", "Change my email",
    "Account got suspended", "I forgot my username", "How to change password",
    "Account shows wrong name", "Need to verify identity",
    "Account settings not saving", "Account locked",

    "How does your service work", "What are business hours",
    "Do you have a mobile app", "What payment methods accepted",
    "Is there a free trial", "What is your return policy",
    "How to use promo code", "Do you offer student discount",
    "How to contact support", "Are my details safe",
]

intents = (
    ['Billing'] * 10 +
    ['Technical'] * 10 +
    ['Shipping'] * 10 +
    ['Account'] * 10 +
    ['General'] * 10
)

priority = [
    'High' if i in ['Billing','Technical'] else
    'Medium' if i in ['Shipping','Account'] else 'Low'
    for i in intents
]

escalate = [
    'Yes' if i in ['Billing','Technical'] else 'No'
    for i in intents
]

df = pd.DataFrame({
    'ticket_id' : range(1, 51),
    'message' : messages,
    'intent' : intents,
    'priority' : priority,
    'escalate' : escalate
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/tickets.csv', index=False)

print("Dataset created! Total tickets:", len(df))
print("\nTickets per intent:")
print(df['intent'].value_counts().to_string())
print("\nFirst 5 rows:")
print(df.head())