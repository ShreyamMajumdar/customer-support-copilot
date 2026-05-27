import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/tickets.csv')
os.makedirs('outputs', exist_ok=True)

print("Total tickets:", len(df))
print(df['intent'].value_counts().to_string())


intent_counts = df['intent'].value_counts()

plt.figure(figsize=(9, 5))
bars = plt.bar(intent_counts.index, intent_counts.values,
               color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'],
               edgecolor='black')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             str(bar.get_height()), ha='center', fontsize=11)
plt.title('Tickets by Intent', fontsize=14)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/chart1_intent.png')
plt.show()
print("Chart 1 done")

priority_counts = df['priority'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(priority_counts.values,
        labels=priority_counts.index,
        colors=['#e74c3c','#f39c12','#2ecc71'],
        autopct='%1.1f%%', startangle=140)
plt.title('Ticket Priority Split', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/chart2_priority.png')
plt.show()
print("Chart 2 done")

escalate_counts = df['escalate'].value_counts()

plt.figure(figsize=(6, 5))
bars = plt.bar(['Needs Human', 'Chatbot Handles'],
               escalate_counts.values,
               color=['#e74c3c','#2ecc71'],
               edgecolor='black', width=0.4)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             str(bar.get_height()), ha='center', fontsize=12)
plt.title('Escalation -- Human vs Chatbot', fontsize=14)
plt.ylabel('Number of Tickets')
plt.tight_layout()
plt.savefig('outputs/chart3_escalation.png')
plt.show()
print("Chart 3 done")