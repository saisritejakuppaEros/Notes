
# 📘 System Design — Introduction

> **Sources:** GeeksforGeeks | Topics: System Design Intro, Functional vs Non-Functional Requirements, HLD vs LLD

---

> [!abstract]- 📚 Table of Contents (click to expand)
> - [[#1. What is System Design?]]
>     - [[#1.1 Role in SDLC]]
>     - [[#1.2 Two Parts of System Design]]
> - [[#2. High-Level Design (HLD)]]
>     - [[#2.1 What HLD Covers]]
>     - [[#2.2 Prerequisite Knowledge for HLD]]
>     - [[#2.3 Real-World HLD Examples]]
> - [[#3. Low-Level Design (LLD)]]
>     - [[#3.1 What LLD Covers]]
>     - [[#3.2 Prerequisite Knowledge for LLD]]
> - [[#4. HLD vs LLD — Side-by-Side Comparison]]
> - [[#5. Functional Requirements (FR)]]
>     - [[#5.1 Definition]]
>     - [[#5.2 Characteristics]]
>     - [[#5.3 How to Gather FRs]]
> - [[#6. Non-Functional Requirements (NFR)]]
>     - [[#6.1 Definition]]
>     - [[#6.2 Key NFR Dimensions]]
>     - [[#6.3 How to Gather NFRs]]
> - [[#7. FR vs NFR — Comparison Table]]
> - [[#8. Real-World FR + NFR Examples]]
> - [[#9. Challenges in Defining Requirements]]
> - [[#10. Steps to Approach a System Design Problem]]
> - [[#11. Interview Strategy: Approaching a Design Problem]]
> - [[#12. Key Quality Attributes Every System Must Have]]

---

## 1. What is System Design?

System design is the process of **planning, structuring, and defining the architecture** of a software system. It translates user requirements into a detailed blueprint that guides the implementation phase.

The core goals are:

- Create a **well-organized, efficient structure**
- Address **scalability** — can the system grow?
- Address **maintainability** — can it be easily updated?
- Address **performance** — does it respond fast enough?

> 💡 Think of system design like an **architect's blueprint** for a building — before anyone lays a brick, the entire structure is mapped out.

---

### 1.1 Role in SDLC

System design sits between **requirements gathering** and **implementation** in the Software Development Life Cycle (SDLC).

```
Requirements → [System Design] → Implementation → Testing → Deployment
```

- Without a designing phase, you **cannot jump to implementation or testing**
- It represents the **business logic backbone** of the software
- It helps handle **exceptional/edge case scenarios** before coding begins

---

### 1.2 Two Parts of System Design

System design is split into two complementary phases:

|Phase|Focus|
|---|---|
|**HLD** (High-Level Design)|Overall architecture — the big picture|
|**LLD** (Low-Level Design)|Internal logic of each component — the blueprint for developers|

---

## 2. High-Level Design (HLD)

HLD lays out the **overall system architecture** — how major components interact, what services or modules exist, and how data flows among them.

- Also called **macro-level design** or **system design**
- Created by **solution architects**, reviewed by stakeholders and managers
- Input: **Software Requirement Specification (SRS)**
- Output: **Database design, functional design, review records**

---

### 2.1 What HLD Covers

|Topic|Description|
|---|---|
|**System architecture overview**|Major components, modules, and their interactions (services, queues, databases)|
|**Data flow & component interaction**|How data moves between modules; key integrations|
|**Technology stack & infrastructure**|Frameworks, platforms, hardware, databases, hosting|
|**Module responsibilities**|What each module does and how they relate|
|**Performance & trade-offs**|Scalability, security, cost, other non-functional factors|
|**Artifacts**|Architecture diagrams, component diagrams, deployment diagrams, data flow diagrams, ER overviews|

---

### 2.2 Prerequisite Knowledge for HLD

To do HLD well, you need:

- **Basic coding skills** (DSA)
- **Databases** — SQL and NoSQL (MySQL, PostgreSQL, MongoDB, Cassandra)
- **Caches** — Redis, Memcached, CDNs
- **APIs** — REST, GraphQL
- **LLD fundamentals** — OOP and Design Patterns
- **Functional & Non-Functional Requirements** — understanding trade-offs
- **Networking & Security** — DNS, TCP/UDP, HTTP, WebSockets, OAuth, JWT, TLS/SSL, rate limiting, DDoS basics
- **Message queues & streaming** — Kafka, RabbitMQ
- **Architecture patterns** — Microservices vs Monoliths, fault tolerance, load balancing algorithms
- **Observability** — Prometheus, Grafana, ELK Stack (Elasticsearch, Logstash, Kibana), PagerDuty

---

### 2.3 Real-World HLD Examples

|Company|HLD Decision|
|---|---|
|**Netflix**|Migrated from monolith to microservices (starting with encoding and UI), completing by 2011 to handle high-load events like holiday seasons|
|**Uber**|Adopted event-driven architecture — ride requests, location updates, fare changes emit events triggering real-time systems (driver matching, billing, dynamic pricing)|
|**Twitter**|Deployed load-balanced architecture with caching of trending topics and tweets to serve millions of users with real-time data flows|

---

## 3. Low-Level Design (LLD)

LLD describes **how each component works internally** — it gives developers a clear, actionable blueprint to build each piece.

- Also called **micro-level design** or **detailed design**
- Created by **senior developers and designers** (after HLD is finalized)
- Input: **Reviewed HLD**
- Output: **Program specification and unit test plan**

---

### 3.1 What LLD Covers

|Topic|Description|
|---|---|
|**Component/module breakdown**|Internal logic for each module — class responsibilities, methods, attributes, interactions|
|**Database schema & structure**|Designing tables, keys, indexes, relationships with SQL/NoSQL refinements|
|**API & interface definitions**|Precise request/response formats, error codes, methods, endpoints|
|**Error handling & validation**|How each module handles invalid inputs, failures, edge cases, logging|
|**Design patterns & SOLID**|Clean, extensible, maintainable code via established patterns|
|**UML and pseudocode**|Class diagrams, sequence diagrams, pseudocode or flowcharts|

---

### 3.2 Prerequisite Knowledge for LLD

- **Basic coding skills** (DSA)
- **Strong OOP concepts** — Encapsulation, Inheritance, Polymorphism, Abstraction
- **Design Patterns** — Creational, Structural, Behavioral
- **SOLID Principles**

---

## 4. HLD vs LLD — Side-by-Side Comparison

|Dimension|High-Level Design (HLD)|Low-Level Design (LLD)|
|---|---|---|
|**What it is**|Overall system design — the big picture|Component-level design — the fine details|
|**Also known as**|Macro-level / System Design|Micro-level / Detailed Design|
|**Purpose**|Describes overall architecture of the application|Describes detailed logic of each module|
|**Detail level**|Brief functionality of each module|Detailed functional logic of the module|
|**Created by**|Solution Architect|Designers and Developers|
|**Participants**|Design team, Review team, Client team|Design team, Operation teams, Implementers|
|**Order**|Created **first**|Created **second** (after HLD)|
|**Input**|Software Requirement Specification (SRS)|Reviewed HLD|
|**Output**|Database design, functional design, review records|Program specification, unit test plan|
|**Focus**|Business requirements → High-level solution|High-level solution → Detailed implementation|

> 🔑 **Memory trick:** HLD = **What** the system does at scale. LLD = **How** each part is built in code.

---

## 5. Functional Requirements (FR)

### 5.1 Definition

Functional requirements define **the specific features and operations** a system must perform to meet business and user needs. They describe **what the system should do** and how it interacts with users or other systems.

### 5.2 Characteristics

- Focus on **system behavior and functionality**
- Represent features that are **directly observable and testable** in the final product
- Drive the **core design and features** of the system
- Documented in **use cases, user stories, or functional specifications**
- Validated through **functional testing** (unit, integration, acceptance tests)

**Common examples:** user authentication, data processing, search functionality, payment handling, report generation

**Key questions to ask:**

- What features should the system include?
- What edge cases must be considered?

---

### 5.3 How to Gather FRs

- **Interviews** — Talk directly to stakeholders or users to understand their needs
- **Surveys** — Distribute questionnaires to gather input at scale
- **Workshops** — Host brainstorming sessions to gather features and feedback

---

## 6. Non-Functional Requirements (NFR)

### 6.1 Definition

Non-functional requirements define **how a system should operate**, focusing on performance, reliability, and user experience rather than specific features. They ensure the system is **efficient, secure, and maintainable** over time.

### 6.2 Key NFR Dimensions

|NFR Type|What it means|
|---|---|
|**Performance**|Speed and responsiveness (e.g., response time < 2 seconds)|
|**Security**|Protection against unauthorized access|
|**Usability**|Ease of use for end users|
|**Reliability**|System stability and availability|
|**Scalability**|Ability to handle growth in users/load|
|**Maintainability**|Ease of updates and bug fixes|
|**Portability**|Ability to run in different environments|

**Key questions to ask:**

- How fast should the system respond?
- How secure must it be?
- How available and reliable should it be?
- Can it handle future growth?

### 6.3 How to Gather NFRs

- **Performance Benchmarks** — Consult IT teams to set load expectations
- **Security Standards** — Consult security experts for data protection best practices
- **Usability Testing** — Test the system to find UX friction points and refine interfaces

---

## 7. FR vs NFR — Comparison Table

|Dimension|Functional Requirements|Non-Functional Requirements|
|---|---|---|
|**Definition**|What the system should do|How the system should perform|
|**Purpose**|System behavior and features|Performance, usability, and quality|
|**Scope**|Actions and operations the system must support|Constraints under which those actions occur|
|**Measurement**|Easy — verify outputs or results|Harder — benchmarks, metrics, SLAs|
|**Impact on Dev**|Drives core design and feature set|Influences architecture and performance optimization|
|**User perspective**|Directly visible; tied to business needs|Shapes user experience indirectly|
|**Documentation**|Use cases, user stories, functional specs|Performance criteria, technical specs, design constraints|
|**Testing**|Functional testing (unit, integration, acceptance)|Performance, security, and usability testing|
|**Dependency**|Defines what must be built|Defines how well the system must operate|

---

## 8. Real-World FR + NFR Examples

### Online Banking System

**Functional Requirements:**

- Users can log in with username and password
- Users can check their account balance
- Users receive notifications after making a transaction

**Non-Functional Requirements:**

- System responds to user actions in **< 2 seconds**
- All transactions must be **encrypted** (compliance with security standards)
- System should handle **100 million users** with minimal downtime

---

### Food Delivery App

**Functional Requirements:**

- Users can browse menus and place an order
- Users can make payments and track orders in real time

**Non-Functional Requirements:**

- Restaurant menu loads in **< 1 second**
- System supports up to **50,000 concurrent orders** during peak hours
- App is intuitive for **first-time users**

---

## 9. Challenges in Defining Requirements

|Challenge|Description|
|---|---|
|**Ambiguity**|Vague or incomplete requirements make it hard to define what the system must do|
|**Changing requirements**|Evolving business goals or user expectations shift requirements during development|
|**Prioritization difficulty**|FRs often take precedence; critical NFRs like security/scalability get overlooked|
|**Measuring NFRs**|FRs are easy to test; NFRs like usability or reliability are harder to quantify|
|**Conflicting requirements**|Some requirements conflict (e.g., stronger security can reduce performance) — requires trade-offs|

> ⚠️ **Balancing act:** A system with great features but poor performance = bad product. Always design with both FR and NFR in mind from day one.

---

## 10. Steps to Approach a System Design Problem

```
1. Understand Requirements
   └─ Gather & analyze business needs from stakeholders, users, documentation

2. Define Architecture
   └─ Identify key components and their interactions (services, APIs, databases)

3. Choose Tech Stack
   └─ Select languages, databases, frameworks, tools based on requirements

4. Design Modules
   └─ Break the system into modules; define responsibilities and data flow

5. Plan for Scalability
   └─ Anticipate load, optimize bottlenecks, use scalable patterns

6. Ensure Security & Privacy
   └─ Identify risks; implement authentication, encryption, data protection

7. Test & Validate
   └─ Write test cases; simulate real-world usage to verify requirements
```

---

## 11. Interview Strategy: Approaching a Design Problem

### Step 1: Break Down the Problem

- Decompose the problem into smaller **services or features**
- Ask the interviewer to clarify the features in scope — **don't assume**
- You are not expected to design everything in an interview

### Step 2: Communicate Your Ideas

- Keep the interviewer **in the loop** throughout
- Discuss your process **out loud** as you think
- Use **flowcharts and diagrams** on the whiteboard
- Explain your reasoning on: scalability, database design, trade-offs

### Step 3: Make Reasonable Assumptions

- State your assumptions explicitly (e.g., number of requests/day, DB calls/month, cache hit rate)
- Keep numbers **reasonable and backed by logic**
- Examples: "I'll assume 1 million DAU" or "I'll assume a 90% cache hit rate"

---

## 12. Key Quality Attributes Every System Must Have

|Attribute|What to Design For|
|---|---|
|**Scalability**|Handle increased loads; scale horizontally or vertically as needed|
|**Performance**|Minimal latency, efficient processing, fast response times|
|**Reliability**|Minimal downtime, system failures, and unexpected crashes|
|**Security**|Prevent unauthorized access; protect sensitive user data|
|**Maintainability**|Easy to update with clear documentation and organized code|
|**Interoperability**|Works seamlessly with other systems via well-defined interfaces|
|**Usability**|User-friendly, intuitive, consistent UI/UX|
|**Cost-effectiveness**|Minimize development and operational costs while meeting requirements|

---

> 📅 _Notes compiled: March 23, 2026 | Sources: GeeksforGeeks System Design Series_