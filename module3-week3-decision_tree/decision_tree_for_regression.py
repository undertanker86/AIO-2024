from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_labels = machine_cpu.target
X_train, X_test, y_train, y_test = train_test_split(
    machine_data, machine_labels,
    test_size=0.2,
    random_state=42)

tree_reg = DecisionTreeRegressor()

tree_reg.fit(X_train, y_train)


y_pred = tree_reg.predict(X_test)
print(y_pred)
mean_squared_error(y_test, y_pred)
