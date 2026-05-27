import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/tickets.csv')

with open('data/model.pkl', 'rb') as f: model = pickle.load(f)
with open('data/vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)

X = vectorizer.transform(df['message'])
y = df['intent']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
acc = accuracy_score(y_test, model.predict(X_test))

auto = df[df['escalate'] == 'No']
human = df[df['escalate'] == 'Yes']
pct_auto = round(len(auto) / len(df) * 100, 1)

print("=" * 50)
print("FINAL REPORT -- Customer Support Copilot")
print("=" * 50)

print("\nTicket Summary:")
print(" Total tickets :", len(df))
print(" Chatbot can handle :", len(auto), f"({pct_auto}%)")
print(" Needs human agent :", len(human))

print("\nIntent Breakdown:")
print(df['intent'].value_counts().to_string())

print("\nModel Accuracy:", round(acc * 100, 1), "%")
print("Workload reduction:", pct_auto, "%")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Customer Support Copilot -- Summary', fontsize=16, fontweight='bold')

counts = df['intent'].value_counts()
axes[0,0].bar(counts.index, counts.values,
              color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'], edgecolor='black')
axes[0,0].set_title('Tickets by Intent')
axes[0,0].set_ylabel('Count')

p = df['priority'].value_counts()
axes[0,1].pie(p.values, labels=p.index,
              colors=['#e74c3c','#f39c12','#2ecc71'],
              autopct='%1.1f%%', startangle=140)
axes[0,1].set_title('Priority Split')

axes[1,0].bar(['Chatbot', 'Human'], [len(auto), len(human)],
              color=['#2ecc71','#e74c3c'], edgecolor='black', width=0.4)
axes[1,0].set_title(f'Chatbot handles {pct_auto}% automatically')
axes[1,0].set_ylabel('Tickets')

axes[1,1].bar(['Accuracy'], [acc * 100],
              color='#3498db', edgecolor='black', width=0.3)
axes[1,1].text(0, acc * 100 + 1,
               str(round(acc * 100, 1)) + "%", ha='center', fontsize=13)
axes[1,1].set_title('Model Accuracy')
axes[1,1].set_ylim(0, 110)
axes[1,1].set_ylabel('%')

plt.tight_layout()
plt.savefig('outputs/chart5_dashboard.png', dpi=150)
plt.show()

print("\nDashboard saved to outputs/chart5_dashboard.png")

print("\n" + "=" * 50)
print("RECOMMENDATIONS:")
print("=" * 50)
print("""
1. Automate Shipping, Account and General tickets
   -- chatbot can handle these without human help

2. Always route Billing and Technical to humans
   -- these are high priority and sensitive

3. Add more training data over time
   -- more data means higher model accuracy

4. Ask customers if reply was helpful after each ticket
   -- use that feedback to improve the chatbot
""")
print("=" * 50)
