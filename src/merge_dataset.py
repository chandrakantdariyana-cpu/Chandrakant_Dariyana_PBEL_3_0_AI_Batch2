import pandas as pd

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
]

dfs = [pd.read_csv(file) for file in files]

combined = pd.concat(dfs, ignore_index=True)

combined.to_csv("CICIDS2017_Final.csv", index=False)

print("Combined successfully!")
print("Rows and columns:", combined.shape)