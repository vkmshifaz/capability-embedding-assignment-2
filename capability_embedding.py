"""
Assignment 2: Vector Embedding for Capability Composition
Includes numerical experiments, CSV outputs, and graphs.

Install:
    python -m pip install numpy scikit-learn matplotlib

Run:
    python capability_embedding.py

Files generated in the same folder:
    capability_similarity.png
    goal_relevance.png
    operational_attributes.png
    experiment_results.csv
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Set
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class State:
    values: Dict[str, Any]


@dataclass
class Goal:
    conditions: Dict[str, Any]


@dataclass
class Capability:
    name: str
    capability_type: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    resources: Set[str] = field(default_factory=set)
    cost: float = 1.0
    reliability: float = 0.95
    available: bool = True
    mechanism: str = ""


CAPABILITIES = [
    Capability(
        name="CreateOrder", capability_type="API",
        inputs={"cart_id": "UUID"}, outputs={"order_id": "UUID"},
        preconditions={"Cart.exists": True, "Cart.item_count_positive": True},
        effects={"Order.exists": True, "Order.status": "CREATED"},
        constraints={"user_authenticated": True},
        resources={"database", "network"}, cost=2.0, reliability=0.99,
        available=True, mechanism="POST /orders"
    ),
    Capability(
        name="MakePayment", capability_type="SERVICE",
        inputs={"order_id": "UUID"}, outputs={"payment_id": "UUID"},
        preconditions={"Order.exists": True, "Order.status": "CREATED"},
        effects={"Payment.status": "SUCCESS"},
        constraints={"transaction_limit_ok": True},
        resources={"payment_gateway", "network"}, cost=3.0, reliability=0.98,
        available=True, mechanism="payment gateway API"
    ),
    Capability(
        name="SendNotification", capability_type="EVENT",
        inputs={"order_id": "UUID"}, outputs={"notification_id": "UUID"},
        preconditions={"Order.exists": True},
        effects={"Notification.sent": True},
        resources={"message_service", "network"}, cost=1.0, reliability=0.97,
        available=True, mechanism="OrderCreated event handler"
    ),
    Capability(
        name="CancelCart", capability_type="FUNCTION",
        inputs={"cart_id": "UUID"}, outputs={},
        preconditions={"Order.exists": False},
        effects={"Cart.cancelled": True},
        resources=set(), cost=0.5, reliability=0.99,
        available=True, mechanism="cancel_cart()"
    ),
    Capability(
        name="CreateOrderDatabase", capability_type="DATABASE",
        inputs={"cart_id": "UUID"}, outputs={"order_id": "UUID"},
        preconditions={"Cart.exists": True, "Cart.item_count_positive": True},
        effects={"Order.exists": True, "Order.status": "CREATED"},
        constraints={"user_authenticated": True},
        resources={"database"}, cost=1.5, reliability=0.96,
        available=True, mechanism="INSERT INTO orders"
    ),
    Capability(
        name="CheckInventory", capability_type="API",
        inputs={"item_id": "UUID"}, outputs={"inventory_available": "bool"},
        effects={"Inventory.checked": True},
        resources={"inventory_service", "network"}, cost=1.0,
        reliability=0.99, available=True, mechanism="GET /inventory"
    ),
]

INITIAL_STATE = State({
    "User.authenticated": True,
    "Cart.exists": True,
    "Cart.item_count_positive": True,
    "Order.exists": False,
    "Order.status": "NONE",
    "Payment.status": "NOT_STARTED",
    "Notification.sent": False,
})

GOAL = Goal({
    "Order.exists": True,
    "Payment.status": "SUCCESS",
    "Notification.sent": True,
})

FEATURES = [
    "Order.exists", "Order.status.CREATED", "Payment.status.SUCCESS",
    "Notification.sent", "Cart.exists", "Cart.item_count_positive",
    "Inventory.checked", "Cart.cancelled", "input.cart_id", "input.order_id",
    "output.order_id", "output.payment_id", "output.notification_id",
    "type.API", "type.DATABASE", "type.GUI", "type.EVENT", "type.FUNCTION",
    "type.COMPUTATION", "type.MESSAGE", "type.SERVICE",
    "resource.database", "resource.payment_gateway", "resource.network",
    "resource.message_service", "resource.inventory_service",
    "constraint.user_authenticated", "constraint.transaction_limit_ok",
    "cost.normalized", "reliability", "availability",
]
FEATURE_INDEX = {feature: index for index, feature in enumerate(FEATURES)}


def set_feature(vector, feature, value=1.0):
    if feature in FEATURE_INDEX:
        vector[FEATURE_INDEX[feature]] = value


def encode_state(state):
    vector = np.zeros(len(FEATURES), dtype=float)
    for key, value in state.values.items():
        if value is True:
            set_feature(vector, key)
        elif key == "Order.status" and value == "CREATED":
            set_feature(vector, "Order.status.CREATED")
        elif key == "Payment.status" and value == "SUCCESS":
            set_feature(vector, "Payment.status.SUCCESS")
    return vector


def encode_goal(goal):
    vector = np.zeros(len(FEATURES), dtype=float)
    for key, value in goal.conditions.items():
        if value is True:
            set_feature(vector, key)
        elif key == "Order.status" and value == "CREATED":
            set_feature(vector, "Order.status.CREATED")
        elif key == "Payment.status" and value == "SUCCESS":
            set_feature(vector, "Payment.status.SUCCESS")
    return vector


def encode_capability(capability):
    vector = np.zeros(len(FEATURES), dtype=float)
    for key, value in capability.effects.items():
        if value is True:
            set_feature(vector, key)
        elif key == "Order.status" and value == "CREATED":
            set_feature(vector, "Order.status.CREATED")
        elif key == "Payment.status" and value == "SUCCESS":
            set_feature(vector, "Payment.status.SUCCESS")
    for name in capability.inputs:
        set_feature(vector, f"input.{name}")
    for name in capability.outputs:
        set_feature(vector, f"output.{name}")
    set_feature(vector, f"type.{capability.capability_type}")
    for resource in capability.resources:
        set_feature(vector, f"resource.{resource}")
    for name, enabled in capability.constraints.items():
        if enabled:
            set_feature(vector, f"constraint.{name}")
    set_feature(vector, "cost.normalized", min(capability.cost / 5.0, 1.0))
    set_feature(vector, "reliability", capability.reliability)
    set_feature(vector, "availability", 1.0 if capability.available else 0.0)
    return vector


def similarity(vector_a, vector_b):
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        return 0.0
    return float(cosine_similarity(
        vector_a.reshape(1, -1), vector_b.reshape(1, -1)
    )[0, 0])


def compatible(first, second):
    for condition, required_value in second.preconditions.items():
        if condition not in first.effects or first.effects[condition] != required_value:
            return False
    for required_input in second.inputs:
        if required_input not in first.outputs:
            return False
    return True


def compose(capabilities):
    if not capabilities:
        raise ValueError("At least one capability is required.")
    links = []
    accumulated_effects = {}
    accumulated_outputs = set()
    valid = True

    for index, capability in enumerate(capabilities):
        if index > 0:
            preconditions_ok = all(
                condition in accumulated_effects
                and accumulated_effects[condition] == value
                for condition, value in capability.preconditions.items()
            )
            inputs_ok = all(name in accumulated_outputs for name in capability.inputs)
            link_ok = preconditions_ok and inputs_ok
            valid = valid and link_ok
            links.append({
                "from": capabilities[index - 1].name,
                "to": capability.name,
                "compatible": link_ok,
            })
        accumulated_effects.update(capability.effects)
        accumulated_outputs.update(capability.outputs.keys())

    vector = np.mean([encode_capability(c) for c in capabilities], axis=0)
    return {
        "name": " -> ".join(c.name for c in capabilities),
        "links": links, "valid_composition": valid, "vector": vector
    }


def goal_relevance(capability, goal):
    return similarity(encode_capability(capability), encode_goal(goal))


def save_csv(filename, rows, headers):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_experiments():
    by_name = {c.name: c for c in CAPABILITIES}
    print("=" * 72)
    print("ASSIGNMENT 2: VECTOR EMBEDDING FOR CAPABILITY COMPOSITION")
    print("=" * 72)

    state_vector = encode_state(INITIAL_STATE)
    goal_vector = encode_goal(GOAL)
    print("\n1. STATE AND GOAL ENCODING")
    print(f"Feature vector dimension: {len(FEATURES)}")
    print(f"Initial state vector non-zero features: {np.count_nonzero(state_vector)}")
    print(f"Goal vector non-zero features: {np.count_nonzero(goal_vector)}")

    print("\n2. EXPERIMENT: CAPABILITY COMPATIBILITY")
    compatibility_pairs = [
        ("CreateOrder", "MakePayment"),
        ("CreateOrder", "CancelCart"),
        ("CreateOrder", "SendNotification"),
        ("CreateOrderDatabase", "MakePayment"),
    ]
    compatibility_rows = []
    for a, b in compatibility_pairs:
        result = compatible(by_name[a], by_name[b])
        compatibility_rows.append({"first": a, "second": b, "compatible": result})
        print(f"{a} -> {b}: {'COMPATIBLE' if result else 'INCOMPATIBLE'}")

    print("\n3. EXPERIMENT: CAPABILITY SIMILARITY")
    similarity_pairs = [
        ("CreateOrder", "CreateOrderDatabase"),
        ("CreateOrder", "MakePayment"),
        ("CreateOrder", "CancelCart"),
    ]
    similarity_rows = []
    for a, b in similarity_pairs:
        score = similarity(encode_capability(by_name[a]), encode_capability(by_name[b]))
        similarity_rows.append({"capability_1": a, "capability_2": b, "cosine_similarity": round(score, 4)})
        print(f"Similarity({a}, {b}) = {score:.3f}")

    print("\n4. EXPERIMENT: COMPOSITION")
    purchase = compose([by_name["CreateOrder"], by_name["MakePayment"], by_name["SendNotification"]])
    print("Composite:", purchase["name"])
    for link in purchase["links"]:
        print(f"  {link['from']} -> {link['to']}: {link['compatible']}")
    print("Valid composition:", purchase["valid_composition"])
    print("Composite vector dimension:", len(purchase["vector"]))

    print("\n5. EXPERIMENT: GOAL RELEVANCE")
    goal_rows = []
    for capability in CAPABILITIES:
        score = goal_relevance(capability, GOAL)
        goal_rows.append({"capability": capability.name, "goal_similarity": round(score, 4)})
        print(f"{capability.name:22s} goal similarity = {score:.3f}")

    print("\n6. EXPERIMENT: OPERATIONAL ATTRIBUTES")
    operational_rows = []
    print(f"{'Capability':22s} {'Cost':>6s} {'Reliability':>12s} {'Available':>10s}")
    for c in CAPABILITIES:
        operational_rows.append({
            "capability": c.name, "cost": c.cost,
            "reliability": c.reliability, "available": c.available
        })
        print(f"{c.name:22s} {c.cost:6.2f} {c.reliability:12.2f} {str(c.available):>10s}")

    save_csv("experiment_results.csv", similarity_rows, ["capability_1", "capability_2", "cosine_similarity"])
    print("\nSaved experiment_results.csv")

    # Graph 1: capability similarity scores
    labels = [f"{r['capability_1']}\nvs\n{r['capability_2']}" for r in similarity_rows]
    values = [r["cosine_similarity"] for r in similarity_rows]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Cosine similarity")
    plt.title("Capability Similarity Comparison")
    plt.tight_layout()
    plt.savefig("capability_similarity.png", dpi=200)
    plt.close()
    print("Saved capability_similarity.png")

    # Graph 2: each capability's similarity to the goal
    plt.figure(figsize=(10, 5))
    plt.bar([r["capability"] for r in goal_rows], [r["goal_similarity"] for r in goal_rows])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Cosine similarity to goal")
    plt.title("Capability Goal Relevance")
    plt.tight_layout()
    plt.savefig("goal_relevance.png", dpi=200)
    plt.close()
    print("Saved goal_relevance.png")

    # Graph 3: operational attributes on comparable separate scales
    names = [r["capability"] for r in operational_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [r["cost"] for r in operational_rows], width=0.4, label="Cost")
    ax.bar(x + 0.2, [r["reliability"] for r in operational_rows], width=0.4, label="Reliability")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Value (different scales)")
    ax.set_title("Capability Cost and Reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig("operational_attributes.png", dpi=200)
    plt.close(fig)
    print("Saved operational_attributes.png")

    print("\nAll experiments completed.")
    print("Note: this is a baseline with hand-designed features.")
    print("Compatibility is checked symbolically; cosine similarity alone does not prove composability.")


if __name__ == "__main__":
    run_experiments()
