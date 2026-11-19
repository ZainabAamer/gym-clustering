import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("gym_members.csv")
X = df.drop(columns=["member_id"])

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
print(X_train.shape[0], X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = list(range(1, 9))
train_spreads = []
test_spreads = []

for k in ks:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)
    train_inertia = kmeans.inertia_
    test_spread = np.sum(np.min(kmeans.transform(X_test_scaled), axis=1) ** 2)
    train_spreads.append(train_inertia)
    test_spreads.append(test_spread)
    print(k, train_inertia, test_spread)

plt.plot(ks, train_spreads, label="Train (231 members)")
plt.plot(ks, test_spreads, label="Test (99 members)")
plt.title("Within-group spread by number of groups (k)")
plt.xlabel("Number of groups (k)")
plt.ylabel("Within-group spread (sum of squared distances, scaled units)")
plt.xticks(ks)
plt.axvline(4, linestyle="--")
plt.legend()
plt.savefig("elbow.png", dpi=200, bbox_inches="tight")

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X_train_scaled)
print(pd.Series(kmeans.labels_).value_counts().sort_index().to_string())

centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns,
).round(1)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(centers)

SEGMENT_NAMES = {
    0: "Newcomers",
    1: "Veteran regulars",
    2: "Floor grinders",
    3: "Class spenders",
}
GETS_CREDIT = {
    0: True,
    1: False,
    2: False,
    3: False,
}


def segment_action(
    age,
    visits_per_month,
    avg_visit_minutes,
    classes_per_month,
    months_member,
    monthly_extras_spend,
):
    row = pd.DataFrame(
        [
            [
                age,
                visits_per_month,
                avg_visit_minutes,
                classes_per_month,
                months_member,
                monthly_extras_spend,
            ]
        ],
        columns=X.columns,
    )
    scaled = scaler.transform(row)
    print(kmeans.transform(scaled)[0])
    cluster = int(kmeans.predict(scaled)[0])
    return cluster, SEGMENT_NAMES[cluster], GETS_CREDIT[cluster]


print(segment_action(30, 4, 42, 1, 5, 10))

all_members = X.copy()
all_members["cluster"] = kmeans.predict(scaler.transform(X))
all_members["credit_net"] = np.where(
    all_members["monthly_extras_spend"] < 20, 5, -10
)

baseline_total = all_members["credit_net"].sum()
v1_total = all_members.loc[
    all_members["cluster"].map(GETS_CREDIT), "credit_net"
].sum()
print(baseline_total, v1_total, v1_total - baseline_total)

for cluster, group in all_members.groupby("cluster"):
    count = len(group)
    share_under_20 = (group["monthly_extras_spend"] < 20).mean()
    avg_extras = group["monthly_extras_spend"].mean()
    print(cluster, count, share_under_20, avg_extras)

print(test_spreads[0], test_spreads[3])
