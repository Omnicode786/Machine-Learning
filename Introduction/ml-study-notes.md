# My ML Learning Notes - Video 1 to 5

## Video 1: What is Machine Learning?

**Date Watched:** March 13, 2021 | Duration: 20:00

### What I learned:

Machine learning is basically about teaching computers to learn from data and improve automatically without being explicitly programmed. It's part of artificial intelligence, and honestly, it's pretty cool how it works.

**Key Takeaways:**
- ML builds models based on training data to make predictions
- Unlike traditional programming where I write explicit rules, ML finds patterns automatically
- Applications are everywhere - medicine, email filtering, speech recognition, computer vision
- Where it's hard to write conventional algorithms, ML steps in and saves the day

**Real-world examples that stuck with me:**
- Medical diagnosis systems
- Email spam filters (honestly, I wonder how they work so well)
- Voice recognition like Siri and Google Assistant
- Image recognition in cameras

**Why this matters:**
The whole point is that sometimes the problem is too complex to code manually. There are too many variables, too many edge cases. That's where ML shines - it learns from examples instead of me trying to write rules for every scenario.

---

## Video 2: AI vs ML vs DL for Beginners

**Date Watched:** March 15, 2021 | Duration: 16:01

### The Hierarchy (This was confusing at first, but now it makes sense)

**AI (Artificial Intelligence) - The Biggest Umbrella:**
- Everything about making machines smarter
- It's the broadest concept

**ML (Machine Learning) - Inside AI:**
- A subset of AI
- Systems that learn from data without explicit programming
- They get smarter over time without human intervention
- This is what I'm focusing on mainly

**DL (Deep Learning) - Inside ML:**
- ML applied to large datasets
- Uses neural networks (mimics how the brain works)
- Requires massive amounts of data to work well
- More complex than regular ML

### My Understanding Now:
Think of it like this:
- AI is the entire field of smart machines
- ML is a specific approach within that field
- DL is a specialized technique within ML

### When I should use each:

**Regular ML:** When I have structured data, good amount of examples, and the problem is relatively well-defined

**Deep Learning:** When I have huge amounts of data (especially images, text, audio) and the patterns are really complex

**Why DL became important:**
- In 2012, DL started crushing traditional ML in competitions
- Better hardware became available
- More data started existing
- I was wondering why DL got so hyped... now I know it just works better for big data problems

**The key difference:**
With regular ML, I have to tell the system what features to look for (like ear size for dog classification). With DL, the system figures out what features matter on its own.

---

## Video 3: Types of Machine Learning

**Date Watched:** March 16, 2021 | Duration: 27:41

### This is foundational stuff. I need to know which type my problem falls into.

**1. SUPERVISED LEARNING (I have labeled data)**

**Regression Subtype:**
- Used when output is numerical
- Example that made sense: Predicting package salary based on IQ and CGPA
- If salary is 4.5L, 3.2L, 3.8L - these are numbers, so it's regression

**Classification Subtype:**
- Used when output is categorical
- Example: Placement - Yes or No (only two categories)
- Like identifying if an image contains a dog or not

**Key insight:** Just check the output column
- If it's numbers → Regression
- If it's categories → Classification

**2. UNSUPERVISED LEARNING (No labels, I have to find patterns)**

**Clustering:**
- Groups similar data together
- Real example from business: The Beer and Diaper Story
- Walmart found that people buying diapers also bought beer
- By putting them together, they increased sales!
- This is SO useful for customer segmentation

**Dimensionality Reduction:**
- Taking 1000+ features and reducing to just 2-3 dimensions
- Still keeps the important patterns
- Helps visualize high-dimensional data
- Makes computation faster

**Anomaly Detection:**
- Finding outliers
- Manufacturing: detecting defects
- Banking: fraud detection
- It's like finding the weird data points

**Association Rule Learning:**
- Finding relationships between items
- Like the famous Beer and Diaper example
- Used everywhere in e-commerce

**3. SEMI-SUPERVISED LEARNING (Some labeled, some not)**

This was interesting. Real example: Google Photos
- I label one person's photo as "Mom"
- System automatically groups all of Mom's photos together
- I label another as "Dad"
- System groups all Dad's photos
- I only had to label a few, but the system did most of the work

**Why use this?** Creating labels is expensive and time-consuming.

**4. REINFORCEMENT LEARNING (Learning through rewards and punishments)**

This is basically how we learn in real life.

**Simple example:** Learning to drive
- You try different actions
- Some actions lead to good outcomes (you drive safely)
- Some lead to bad outcomes (you hit something)
- You learn which actions are good and which are bad

**In ML terms:**
- Agent: The learning entity (like a robot or AI)
- Environment: The world it operates in
- Action: What the agent does
- Reward/Punishment: Feedback about whether it was good or bad
- Policy: The rules the agent learns

**Famous example:** AlphaGo
- Google's AI that beat a professional Go player
- It wasn't explicitly programmed to play Go
- It learned through millions of games against itself
- Got rewarded for winning, punished for losing

**My takeaway:** This is powerful but complex. Not for every problem.

---

## Video 4: Batch Machine Learning (Offline Learning)

**Date Watched:** March 17, 2021 | Duration: 11:28

### This is about HOW my model learns in production

**What is Batch/Offline Learning?**

The conventional approach:
1. I have all my training data
2. I train the model on my local machine (development environment)
3. When it's ready, I deploy it to a server
4. The model sits there and makes predictions
5. If I need to update it, I have to repeat the entire process

**Real-world flow:**
1. Data scientist trains model locally (costs money, takes time)
2. Model learns patterns from the data
3. Deploy trained model to production
4. Server uses the fixed model for predictions
5. Model doesn't improve from new data automatically

### Advantages I can see:
- Simpler to implement
- Easy to understand and debug
- Works well for problems that don't change much

### Disadvantages (This is important!):

**1. Hardware Limitations:**
If my dataset is really big, my computer might not have enough RAM to load all the data at once. The training process is computationally expensive and time-consuming.

**2. Model Becomes Outdated:**
This hit me hard - once deployed, the model doesn't adapt to new patterns. Like if I built a recommendation system for Netflix last year, new shows and user preferences won't be automatically learned.

**Example:** If my email spam filter was trained in 2020, it might struggle with new spam techniques in 2024. I'd have to manually retrain it.

**3. No Real-time Adaptation:**
If the business scenario changes (like a new trend), my model is stuck with old knowledge.

**4. Connectivity Issues:**
If the model is deployed on a device with no internet access (like a satellite), I can't update it frequently.

**When to use:** Stable problems where patterns don't change much, or when you have computational power to train on big datasets.

---

## Video 5: Online Machine Learning (Incremental Learning)

**Date Watched:** March 18, 2021 | Duration: 19:27

### This blew my mind - continuous learning during production!

**What is Online Learning?**

Instead of training once and deploying, the model keeps learning from new data continuously:
1. Start with initial training data
2. Deploy the model
3. As new data comes in, keep updating the model
4. The model improves over time with real usage

**Key difference from Batch:**
- Batch: Train once → Deploy → Static
- Online: Deploy → Continuously train as new data arrives → Evolving

### Real-world Examples That Made Sense:

**Chatbots (ChatGPT, Alexa, Google Assistant):**
- They learn from conversations happening right now
- As people interact, the model gets better

**Keyboard Autocorrect (SwiftKey):**
- Remember how it learns your typing habits?
- As I type, it's learning my patterns
- That's online learning in action

**YouTube Recommendations:**
- After I watch a video, my recommendations change
- The system learns what I like in real-time
- Not waiting for a batch update

**Email Spam Filters:**
- They adapt to new spam techniques continuously
- Not waiting for me to retrain and redeploy

### How it works technically:

1. I have a trained model initially
2. New data arrives continuously
3. Feed small batches of new data to the model
4. Model updates with partial_fit() or similar
5. Predictions improve gradually

### Learning Rate - This is Critical!

The learning rate determines how much the new data influences the model:
- **Too high:** Model forgets old patterns too quickly (catastrophic forgetting)
- **Too low:** Model is too slow to adapt to new patterns
- **Just right:** Remembers the past but learns new things

It's like me learning - I shouldn't completely forget everything I knew just because I learned something new, but I should also adapt.

### Libraries I Can Use:

**River:** 
- Python library specifically for online ML
- Modern and clean API
- Great for streaming data

**Vowpal Wabbit:**
- More enterprise-grade
- Used in industry
- More complex but powerful

**Scikit-learn's partial_fit():**
- I can use regular scikit-learn models with incremental learning
- SGDClassifier and SGDRegressor support this
- Limited but works for many cases

### Out-of-Core Learning:

If my dataset is too big to fit in RAM:
1. Split data into chunks
2. Train on chunk 1
3. Train on chunk 2 (keeping what I learned from chunk 1)
4. Continue for all chunks

**Advantage:** I can process datasets larger than my computer's memory!

### Disadvantages (Reality check):

**1. Monitoring Complexity:**
The hardest part isn't building the model - it's monitoring and maintaining it in production. I need to constantly check:
- Is the model still performing well?
- Are new patterns emerging?
- Is something broken?

**2. Catastrophic Interference/Forgetting:**
If bad data comes in, the model learns bad patterns and forgets good ones. Can't easily undo this.

**3. Malicious Data Risk:**
If someone hacks the server and sends bad data, my model corrupts itself.

**4. Security Concerns:**
- Need anomaly detection to catch bad data
- Need ability to rollback to previous model versions
- Need monitoring 24/7

**5. Data Dependency:**
Model quality depends entirely on incoming data quality. No human validation step like in batch learning.

### When should I use Online Learning?

✅ **USE IT when:**
- Data changes rapidly (financial markets, social media trends)
- I need cost-effective training (don't retrain on all data)
- The problem itself evolves over time
- I have streaming data coming continuously
- I want recommendations/predictions to improve with usage

❌ **DON'T USE IT when:**
- The problem is stable and doesn't change
- I have good computational resources for batch learning
- There's no continuous data stream
- I can't handle monitoring complexity

### Batch vs Online - Final Comparison:

**Batch Learning:**
- Train once on all data
- Deploy static model
- Simple to implement
- Outdates over time
- Cheaper computationally for one-time training

**Online Learning:**
- Continuous incremental training
- Model adapts automatically
- Complex monitoring needed
- Always up-to-date
- Better for evolving problems
- Cheaper in terms of computational resources per update

---

## Overall Insights After These 5 Videos:

1. **Foundation is crucial:** Understanding what ML is and its types helps me choose the right approach

2. **Real-world context matters:** Whether I use batch or online learning depends on the actual business problem, not just what's theoretically better

3. **There's no free lunch:** Every approach has tradeoffs
   - Batch = simpler but static
   - Online = adaptive but complex monitoring

4. **I need to think like an engineer:** Not just "will it work?" but also "can I maintain it?" and "what happens when things go wrong?"

5. **This is foundational knowledge:** All the algorithms I learn later will fit into these categories and deployment strategies

---

## What I'm Excited To Learn Next:

- How to actually build these models
- Feature engineering (making data useful)
- Evaluation metrics (how do I know if my model is good?)
- Specific algorithms (regression, classification, etc.)
- How companies actually deploy and maintain ML systems

## Questions I Still Have:

- How exactly does partial_fit work under the hood?
- What are real examples of companies using online learning?
- How do I monitor a deployed ML model in practice?
- What's the difference between different ML frameworks?

---

**Last Updated:** While watching 100 Days of ML Playlist
**Next Videos:** Probably going deeper into specific algorithms and hands-on implementation