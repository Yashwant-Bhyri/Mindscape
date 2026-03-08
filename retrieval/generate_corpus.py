import json
import os

corpus = {
    "mood": [
        {"title": "Bipolar II Disorder Criteria", "condition": "Bipolar II Disorder", "content": "Criteria for at least one hypomanic episode and at least one major depressive episode. There has never been a manic episode. Hypomanic episode: Lasting at least 4 consecutive days, with 3+ symptoms (inflated self-esteem, decreased sleep, talkative, racing thoughts, distractibility)."},
        {"title": "Persistent Depressive Disorder (Dysthymia)", "condition": "Dysthymia", "content": "Depressed mood for most of the day, for more days than not, for at least 2 years. Presence of 2+: Poor appetite or overeating, insomnia or hypersomnia, low energy, low self-esteem, poor concentration, feelings of hopelessness."},
        {"title": "Cyclothymic Disorder", "condition": "Cyclothymic Disorder", "content": "For at least 2 years, numerous periods with hypomanic symptoms that do not meet criteria for hypomanic episode and numerous periods with depressive symptoms that do not meet criteria for MDD. Symptoms present at least half the time."}
    ],
    "anxiety": [
        {"title": "Panic Disorder Criteria", "condition": "Panic Disorder", "content": "Recurrent unexpected panic attacks. At least one attack followed by 1 month+ of: Persistent concern about additional attacks, or a significant maladaptive change in behavior related to attacks. Panic attack symptoms: Palpitations, sweating, trembling, shortness of breath, chest pain, nausea, dizziness, chills or heat sensations."},
        {"title": "Social Anxiety Disorder (Social Phobia)", "condition": "Social Anxiety Disorder", "content": "Marked fear or anxiety about one or more social situations in which the individual is exposed to possible scrutiny by others. Fear of acting in a way or showing anxiety symptoms that will be negatively evaluated. Situations almost always provoke fear/anxiety."},
        {"title": "Agoraphobia", "condition": "Agoraphobia", "content": "Marked fear or anxiety about 2+: Using public transport, being in open spaces, being in enclosed places, standing in line/crowds, being outside of home alone. Individual fears escape might be difficult or help unavailable."},
        {"title": "Separation Anxiety Disorder", "condition": "Separation Anxiety Disorder", "content": "Developmentally inappropriate and excessive fear or anxiety concerning separation from those to whom the individual is attached. Persistent worry about losing attachment figures or major harm to them."},
        {"title": "Specific Phobia", "condition": "Specific Phobia", "content": "Marked fear or anxiety about a specific object or situation (e.g., flying, heights, animals, receiving an injection, seeing blood). The phobic object or situation almost always provokes immediate fear or anxiety and is actively avoided or endured with intense fear."},
        {"title": "Selective Mutism", "condition": "Selective Mutism", "content": "Consistent failure to speak in specific social situations in which there is an expectation for speaking (e.g., at school) despite speaking in other situations. The disturbance interferes with educational or occupational achievement or with social communication."}
    ],
    "trauma": [
        {"title": "Posttraumatic Stress Disorder (PTSD)", "condition": "PTSD", "content": "Exposure to actual or threatened death, serious injury, or sexual violence. Intrusion symptoms: Distressing memories, dreams, dissociative reactions (flashbacks). Avoidance of stimuli associated with the event. Negative alterations in cognitions and mood. Marked alterations in arousal and reactivity (hypervigilance, exaggerated startle)."},
        {"title": "Adjustment Disorder", "condition": "Adjustment Disorder", "content": "Development of emotional or behavioral symptoms in response to an identifiable stressor occurring within 3 months of onset of stressor. Distress is out of proportion to the severity of the stressor or results in significant impairment."},
        {"title": "Acute Stress Disorder", "condition": "Acute Stress Disorder", "content": "Exposure to actual or threatened death, serious injury, or sexual violence. Presence of 9+ symptoms from: Intrusion, negative mood, dissociation, avoidance, and arousal, beginning or worsening after the traumatic event. Duration: 3 days to 1 month after trauma."},
        {"title": "Reactive Attachment Disorder", "condition": "Reactive Attachment Disorder", "content": "A consistent pattern of inhibited, emotionally withdrawn behavior toward adult caregivers, manifested by the child rarely or minimally seeking or responding to comfort when distressed. Persistent social and emotional disturbance (minimal social responsiveness, limited positive affect, episodes of unexplained irritability/sadness/fearfulness)."}
    ],
    "psychotic": [
        {"title": "Schizophrenia Criteria", "condition": "Schizophrenia", "content": "Two or more of the following, each present for a significant portion of time during a 1-month period: 1. Delusions, 2. Hallucinations, 3. Disorganized speech, 4. Grossly disorganized or catatonic behavior, 5. Negative symptoms. Continuous signs of disturbance for at least 6 months."},
        {"title": "Schizoaffective Disorder", "condition": "Schizoaffective Disorder", "content": "An uninterrupted period of illness during which there is a major mood episode (manic or depressive) concurrent with Criterion A of schizophrenia. Delusions or hallucinations for 2 or more weeks in the absence of a major mood episode."},
        {"title": "Delusional Disorder", "condition": "Delusional Disorder", "content": "The presence of one or more delusions with a duration of 1 month or longer. Criterion A for schizophrenia has never been met. Functioning is not markedly impaired, and behavior is not obviously bizarre or odd."},
        {"title": "Brief Psychotic Disorder", "condition": "Brief Psychotic Disorder", "content": "Presence of one or more: 1. Delusions, 2. Hallucinations, 3. Disorganized speech, 4. Grossly disorganized or catatonic behavior. Duration is at least 1 day but less than 1 month, with eventual full return to premorbid level of functioning."},
        {"title": "Schizophreniform Disorder", "condition": "Schizophreniform Disorder", "content": "Two or more: Delusions, hallucinations, disorganized speech, behavior, negative symptoms. Duration at least 1 month but less than 6 months."}
    ],
    "personality": [
        {"title": "Borderline Personality Disorder (BPD)", "condition": "BPD", "content": "Pervasive pattern of instability of interpersonal relationships, self-image, and affects, and marked impulsivity. 5+: Frantic efforts to avoid abandonment, unstable/intense relationships, identity disturbance, impulsivity, suicidal behavior/self-mutilation, affective instability, chronic feelings of emptiness, inappropriate anger, paranoid ideation."},
        {"title": "Antisocial Personality Disorder", "condition": "Antisocial Personality Disorder", "content": "Pervasive pattern of disregard for and violation of the rights of others, occurring since age 15. 3+: Failure to conform to social norms/laws, deceitfulness, impulsivity, irritability/aggressiveness, reckless disregard for safety, irresponsibility, lack of remorse."},
        {"title": "Narcissistic Personality Disorder", "condition": "Narcissistic Personality Disorder", "content": "Pervasive pattern of grandiosity, need for admiration, and lack of empathy. 5+: Grandiose self-importance, fantasies of unlimited success/power, belief of being 'special', requires excessive admiration, sense of entitlement, exploitative of others, envious of others."},
        {"title": "Avoidant Personality Disorder", "condition": "Avoidant Personality Disorder", "content": "Pervasive pattern of social inhibition, feelings of inadequacy, and hypersensitivity to negative evaluation. 4+: Avoids occupational activities involving significant interpersonal contact, unwilling to get involved unless certain of being liked, restraint in intimate relationships, preoccupied with being criticized."},
        {"title": "Histrionic Personality Disorder", "condition": "Histrionic Personality Disorder", "content": "Pervasive pattern of excessive emotionality and attention seeking. uncomfortable in situations in which he or she is not the center of attention; interaction with others is often characterized by inappropriate sexually seductive or provocative behavior."},
        {"title": "Schizotypal Personality Disorder", "condition": "Schizotypal Personality Disorder", "content": "Pervasive pattern of social and interpersonal deficits marked by acute discomfort with, and reduced capacity for, close relationships as well as by cognitive or perceptual distortions and eccentricities of behavior."},
        {"title": "Obsessive-Compulsive Personality Disorder (OCPD)", "condition": "OCPD", "content": "Pervasive pattern of preoccupation with orderliness, perfectionism, and mental and interpersonal control, at the expense of flexibility, openness, and efficiency."}
    ],
    "neurodevelopmental": [
        {"title": "ADHD Combined Presentation", "condition": "ADHD", "content": "Inattention: 6+ symptoms (fails to give close attention, difficulty sustaining attention, does not seem to listen, fails to follow instructions, difficulty organizing, avoids sustained mental effort, loses things, distracted, forgetful). Hyperactivity/Impulsivity: 6+ symptoms (fidgets, leaves seat, runs/climbs, unable to play quietly, 'on the go', talks excessively, blurts out answers, difficulty waiting turn, interrupts)."},
        {"title": "Autism Spectrum Disorder (ASD)", "condition": "ASD", "content": "Persistent deficits in social communication and social interaction across multiple contexts. Restricted, repetitive patterns of behavior, interests, or activities (stereotyped movements, insistence on sameness, highly restricted fixated interests, hyper- or hyporeactivity to sensory input)."},
        {"title": "Tourette's Disorder", "condition": "Tourette's Disorder", "content": "Both multiple motor and one or more vocal tics have been present at some time during the illness, although not necessarily concurrently. The tics may wax and wane in frequency but have persisted for more than 1 year since first tic onset."},
        {"title": "Intellectual Disability (Intellectual Developmental Disorder)", "condition": "Intellectual Disability", "content": "Deficits in intellectual functions, such as reasoning, problem solving, planning, abstract thinking, judgment, academic learning, and learning from experience, confirmed by both clinical assessment and individualized, standardized intelligence testing."},
        {"title": "Specific Learning Disorder", "condition": "Specific Learning Disorder", "content": "Difficulties learning and using academic skills, as indicated by the presence of at least one of the following symptoms that have persisted for at least 6 months, despite the provision of interventions that target those difficulties: 1. Inaccurate or slow and effortful word reading; 2. Difficulty understanding the meaning of what is read; 3. Difficulties with spelling; 4. Difficulties with written expression; 5. Difficulties mastering number sense; 6. Difficulties with mathematical reasoning."}
    ],
    "obsessive_compulsive": [
        {"title": "Obsessive-Compulsive Disorder (OCD)", "condition": "OCD", "content": "Presence of obsessions, compulsions, or both. Obsessions: Recurrent/persistent thoughts, urges, or images experienced as intrusive and unwanted, causing anxiety/distress. Compulsions: Repetitive behaviors or mental acts that the individual feels driven to perform in response to an obsession."},
        {"title": "Body Dysmorphic Disorder", "condition": "Body Dysmorphic Disorder", "content": "Preoccupation with one or more perceived defects or flaws in physical appearance that are not observable or appear slight to others. Repetitive behaviors (mirror checking, excessive grooming, skin picking) or mental acts in response to appearance concerns."},
        {"title": "Hoarding Disorder", "condition": "Hoarding Disorder", "content": "Persistent difficulty discarding or parting with possessions, regardless of their actual value. This difficulty is due to a perceived need to save the items and to distress associated with discarding them."}
    ],
    "eating": [
        {"title": "Anorexia Nervosa", "condition": "Anorexia Nervosa", "content": "Restriction of energy intake relative to requirements, leading to a significantly low body weight. Intense fear of gaining weight or becoming fat. Disturbance in the way in which one's body weight or shape is experienced."},
        {"title": "Bulimia Nervosa", "condition": "Bulimia Nervosa", "content": "Recurrent episodes of binge eating. Recurrent inappropriate compensatory behaviors to prevent weight gain (self-induced vomiting, misuse of laxatives, fasting, excessive exercise). Binge eating and compensatory behaviors occur at least once a week for 3 months."},
        {"title": "Binge-Eating Disorder", "condition": "Binge-Eating Disorder", "content": "Recurrent episodes of binge eating. Binge-eating episodes associated with 3+: Eating more rapidly, eating until uncomfortably full, eating large amounts when not hungry, eating alone because of embarrassment, feeling disgusted/guilty afterwards."}
    ],
    "sleep": [
        {"title": "Insomnia Disorder", "condition": "Insomnia Disorder", "content": "Predominant complaint of dissatisfaction with sleep quantity or quality, associated with 1+: Difficulty initiating sleep, difficulty maintaining sleep, early-morning awakening. Occurs at least 3 nights per week for at least 3 months."},
        {"title": "Narcolepsy", "condition": "Narcolepsy", "content": "Recurrent periods of an irrepressible need to sleep, lapsing into sleep, or napping occurring within the same day. These must have been occurring at least three times per week over the past 3 months."}
    ],
    "substance": [
        {"title": "Alcohol Use Disorder", "condition": "Alcohol Use Disorder", "content": "A problematic pattern of alcohol use leading to clinically significant impairment or distress, with 2+: Alcohol taken in larger amounts/over longer period than intended, persistent desire/unsuccessful efforts to cut down, great deal of time spent obtaining/using/recovering, craving, failure to fulfill obligations, continued use despite social/interpersonal problems."},
        {"title": "Opioid Use Disorder", "condition": "Opioid Use Disorder", "content": "Problematic pattern of opioid use with 2+: Opioids taken in larger amounts than intended, persistent desire to cut down, time spent obtaining/using, craving, social/occupational impairment, withdrawal symptoms, tolerance."},
        {"title": "Cannabis Use Disorder", "condition": "Cannabis Use Disorder", "content": "Problematic pattern of cannabis use with 2+: Cannabis taken in larger amounts than intended, desire to cut down, craving, usage in hazardous situations, continued use despite knowledge of physical/psychological problems."}
    ],
    "dissociative": [
        {"title": "Dissociative Identity Disorder (DID)", "condition": "DID", "content": "Disruption of identity characterized by two or more distinct personality states. The disruption involves marked discontinuity in sense of self and sense of agency, accompanied by related alterations in affect, behavior, consciousness, memory, perception."},
        {"title": "Dissociative Amnesia", "condition": "Dissociative Amnesia", "content": "An inability to recall important autobiographical information, usually of a traumatic or stressful nature, that is inconsistent with ordinary forgetting."}
    ],
    "somatic": [
        {"title": "Somatic Symptom Disorder", "condition": "Somatic Symptom Disorder", "content": "One or more somatic symptoms that are distressing or result in significant disruption of daily life. Excessive thoughts, feelings, or behaviors related to the somatic symptoms (disproportionate thoughts, high anxiety about health, excessive time/energy devoted to symptoms)."},
        {"title": "Illness Anxiety Disorder", "condition": "Illness Anxiety Disorder", "content": "Preoccupation with having or acquiring a serious illness. Somatic symptoms are not present or, if present, are only mild in intensity. High level of anxiety about health, performs excessive health-related behaviors."}
    ]
}

# Source mapping
for cat, docs in corpus.items():
    for doc in docs:
        doc["source"] = "DSM-5-TR"
        
# Add some extra placeholders to reach 50 if needed
# (I'll just add more common ones to 'other' or existing cats)
corpus["other"] = [
    {"title": "Gender Dysphoria", "condition": "Gender Dysphoria", "content": "A marked incongruence between one’s experienced/expressed gender and assigned gender, of at least 6 months’ duration, as manifested by at least 2: Marked incongruence between experienced gender and primary/secondary sex characteristics, strong desire to be rid of sex characteristics, strong desire for the sex characteristics of the other gender."},
    {"title": "Premenstrual Dysphoric Disorder (PMDD)", "condition": "PMDD", "content": "In most menstrual cycles, at least 5 symptoms must be present in the final week before the onset of menses, start to improve within a few days after the onset of menses, and become minimal or absent in the week postmenses: Affective lability, irritability, depressed mood, anxiety, decreased interest, difficulty concentrating, lethargy."},
    {"title": "Intermittent Explosive Disorder", "condition": "Intermittent Explosive Disorder", "content": "Recurrent behavioral outbursts representing a failure to control aggressive impulses. Verbal aggression or physical aggression toward property/animals/others occurring twice weekly for 3 months, or 3 outbursts involving damage/injury within 12 months."},
    {"title": "Gambling Disorder", "condition": "Gambling Disorder", "content": "Persistent and recurrent problematic gambling behavior leading to clinically significant impairment or distress. 4+: Needs to gamble with increasing amounts of money, restless/irritable when attempting to cut down, repeated unsuccessful efforts to stop, preoccupied with gambling."},
    {"title": "Factitious Disorder", "condition": "Factitious Disorder", "content": "Falsification of physical or psychological signs or symptoms, or induction of injury or disease, associated with identified deception. The individual presents himself or herself to others as ill, impaired, or injured. The deceptive behavior is evident even in the absence of obvious external rewards."}
]

def save_corpus():
    count = 0
    for category, docs in corpus.items():
        dir_path = f"data/corpus/{category}"
        os.makedirs(dir_path, exist_ok=True)
        for doc in docs:
            filename = doc["condition"].lower().replace("/", "_").replace(" ", "_") + ".json"
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            count += 1
    print(f"Generated {count} clinical documents.")

if __name__ == "__main__":
    save_corpus()
