import random
import csv

safety_templates = [
    "The trial reported severe adverse events including {event}.",
    "Patients experienced grade 3 toxicity such as {event}.",
    "Unexpected mortality was observed due to {event}.",
    "Significant safety concerns were raised after cases of {event}.",
    "High incidence of serious adverse reactions like {event}."
]

dropout_templates = [
    "A high discontinuation rate of {percent}% was observed.",
    "{percent}% of patients withdrew due to intolerable side effects.",
    "Poor treatment compliance led to dropout in {percent}% of cases.",
    "The study reported early withdrawal in {percent}% participants.",
    "Patient retention was low with {percent}% discontinuation."
]

efficacy_templates = [
    "The trial failed to show significant improvement in {endpoint}.",
    "No statistically significant difference was observed in {endpoint}.",
    "The primary endpoint {endpoint} was not met.",
    "Treatment did not improve {endpoint} compared to placebo.",
    "Lack of efficacy was evident in {endpoint} outcomes."
]

normal_templates = [
    "The trial met all primary and secondary endpoints successfully.",
    "Treatment was well tolerated with minimal adverse effects.",
    "The study demonstrated significant improvement in overall survival.",
    "High patient compliance and positive safety profile were observed.",
    "Results confirmed the treatment's effectiveness and safety."
]

events = ["hepatotoxicity", "cardiac arrest", "severe infection", "renal failure", "neurological complications"]
endpoints = ["overall survival", "tumor response rate", "progression-free survival", "clinical remission", "symptom improvement"]

def generate_samples(templates, label, count):
    samples = []
    for _ in range(count):
        template = random.choice(templates)

        if "{event}" in template:
            text = template.format(event=random.choice(events))
        elif "{percent}" in template:
            text = template.format(percent=random.randint(20, 60))
        elif "{endpoint}" in template:
            text = template.format(endpoint=random.choice(endpoints))
        else:
            text = template

        samples.append([text, label])
    return samples


def main():
    dataset = []

    dataset += generate_samples(safety_templates, "Safety Risk", 75)
    dataset += generate_samples(dropout_templates, "Dropout Risk", 75)
    dataset += generate_samples(efficacy_templates, "Efficacy Concern", 75)
    dataset += generate_samples(normal_templates, "Normal Trial", 75)

    random.shuffle(dataset)

    with open("clinical_risk_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)

    print("Dataset generated: clinical_risk_dataset.csv")


if __name__ == "__main__":
    main()
