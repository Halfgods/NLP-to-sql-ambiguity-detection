# Held-out classifier label rationale

One line per pair, authoring-time only (not part of the eval contract).

### CLEAR | Which student has the highest average marks?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Single marks metric named; max over average_marks is the only reading (unique max 96.0).

### CLEAR | Give the names of students who scored above 90 in assignments.

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Literal column (assignment_score) + literal threshold; 7 rows > 90, no boundary student.

### CLEAR | Who owes the most in pending fees?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: pending_fees is the only money-owed concept; ORDER BY DESC LIMIT 1, unique max 10000.

### CLEAR | List the students who have pending fees.

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Pending fees maps only to pending_fees; 'have' = pending_fees > 0 (8 rows). NOT a missing-threshold case: the concept itself is the predicate.

### CLEAR | How many students are there in Computer Engineering?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Department filter + COUNT; no other plausible grouping or metric.

### CLEAR | What is the average pending fees of all students?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Scalar AVG over the single fees column.

### CLEAR | Give the top 3 students by assignment score.

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Named metric, named k; unique values at the cut so deterministic.

### CLEAR | Which students have attendance below 70 and pending fees above 5000?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: Two literal thresholds, one match (Vivaan Sharma).

### CLEAR | How many students are in each department?

- **schema**: `students_college`
- **label**: CLEAR
- **why**: COUNT GROUP BY department; counting people, one table.

### AMBIGUOUS | Which department performs the best overall?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'performs best overall' has 3 column-supported readings: AVG(average_marks), AVG(attendance_percentage), AVG(assignment_score) per department - different queries, different ratios (current data coincidentally agrees on EE; ambiguity is judged on schema, not rescued by data coincidence).

### AMBIGUOUS | Which students are struggling and need support?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: Fork: low marks vs low attendance (vs low assignment); the column choice IS the open decision even though a threshold is also missing (fork dominates per rule).

### AMBIGUOUS | Who are the top performers in internal marks?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'internal marks' has no exact column; it could map to average_marks or assignment_score (internal assessment). Keyword 'marks' looks like a clean map but is a genuine fork.

### AMBIGUOUS | Which students should be considered for the merit scholarship?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: Merit is undefined: rank by average_marks, assignment_score, or attendance - materially different top sets on this data.

### AMBIGUOUS | Which student has the highest overall performance?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'overall performance' unbound across the three metric columns; answers differ (Isha by marks, Ananya by assignments).

### AMBIGUOUS | Which department should get the remedial program?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'Needs it' could be lowest avg marks, lowest avg attendance, or lowest avg assignment - the three pick different depts (IT vs CE).

### AMBIGUOUS | How is each department doing overall?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: Dept-level status unbound across the three metric columns; no single metric named.

### AMBIGUOUS | Give me the weak students for each department.

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'Weak' needs both a metric (marks vs attendance) and a cutoff; metric fork is column-supported and materially different.

### AMBIGUOUS | Who is the brightest student in each department?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'Brightest' can be read as average_marks or assignment_score per department; top sets differ (EE: Diya by marks vs Ananya by assignments).

### AMBIGUOUS | Which department has the highest score?

- **schema**: `students_college`
- **label**: AMBIGUOUS
- **why**: 'score' without a qualifier spreads over average_marks / assignment_score / attendance_percentage.

### UNDERSPECIFIED | Which students are on academic probation?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No probation criteria, flags, or policy table anywhere in the schema.

### UNDERSPECIFIED | How many students live in the hostel?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No hostel/residence column; concept entirely absent (schema cannot answer).

### UNDERSPECIFIED | What is the average age of students by department?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No age or date-of-birth column; unanswerable on this schema.

### UNDERSPECIFIED | Which semester are the current students in?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No semester/year/enrollment-term columns.

### UNDERSPECIFIED | List the phone numbers of the Information Technology students.

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No contact columns in the schema.

### UNDERSPECIFIED | What percentage of students have cleared all assignments?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: 'Cleared' requires a pass cutoff that is not in the schema; single-concept missing parameter, not a column fork.

### UNDERSPECIFIED | Which students have backlogs?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No backlog/course-outcome columns; concept absent.

### UNDERSPECIFIED | What is the grading scale used to compute average_marks?

- **schema**: `students_college`
- **label**: UNDERSPECIFIED
- **why**: No grade-scale metadata; scale lives outside the table data.

### CLEAR | Who earns the most overall?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: Salary-variant CLEAR side: salary is the only income metric, so 'earns the most' has exactly one reading.

### CLEAR | List the employees in Finance with salary above 50000.

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: Literal dept filter + literal threshold on the only money column.

### CLEAR | What is the total salary expense for each department?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: SUM(salary) GROUP BY department; single expense concept.

### CLEAR | Who are the five lowest paid employees?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: Named k, single salary metric.

### CLEAR | How many employees earn below 40000?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: Literal threshold on salary.

### CLEAR | What is the average salary across the company?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: Scalar AVG over salary.

### CLEAR | Which department pays its staff the best on average?

- **schema**: `employees_single_metric`
- **label**: CLEAR
- **why**: 'Pays best on average' = AVG(salary) per department; no competing pay column.

### UNDERSPECIFIED | Are any employees earning more than their team leads?

- **schema**: `employees_single_metric`
- **label**: UNDERSPECIFIED
- **why**: No manager/team-lead column or table; relationship absent.

### UNDERSPECIFIED | Which employees were promoted in the last year?

- **schema**: `employees_single_metric`
- **label**: UNDERSPECIFIED
- **why**: No promotion history or dates.

### UNDERSPECIFIED | How many female employees earn above the company median?

- **schema**: `employees_single_metric`
- **label**: UNDERSPECIFIED
- **why**: No gender column; subgroup cannot be formed.

### UNDERSPECIFIED | Show the annual pay for all married employees.

- **schema**: `employees_single_metric`
- **label**: UNDERSPECIFIED
- **why**: No marital-status column.

### AMBIGUOUS | Who earns the most overall?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: Salary-variant AMBIGUOUS side: income could be base_salary, stock_value_usd, or base+bonus+stock total; materially different top employees possible.

### CLEAR | List the employees whose bonus percentage is above 15.

- **schema**: `employees_multi_metric`
- **label**: CLEAR
- **why**: Literal column + literal threshold.

### CLEAR | Which employee has the highest stock value?

- **schema**: `employees_multi_metric`
- **label**: CLEAR
- **why**: stock_value_usd named directly.

### CLEAR | Compare departments by total travel reimbursement.

- **schema**: `employees_multi_metric`
- **label**: CLEAR
- **why**: travel_reimbursement named; SUM GROUP BY department.

### AMBIGUOUS | Who are the top performers by total compensation?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: 'Total compensation' def not fixed: base only, base+bonus stock equivalent, with or without reimbursements - different orderings.

### AMBIGUOUS | Which department has the highest paid staff?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: Highest PAID staff: avg base vs avg total comp vs max base; different depts can win under each.

### AMBIGUOUS | Who received the most benefits?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: 'Benefits' could be bonus, stock_value_usd, or travel_reimbursement; three columns, materially different winners.

### AMBIGUOUS | Who has the fattest pay package?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: Colloquial 'package' reads as CTC (base+bonus+stock) or just base; fork is column-supported.

### AMBIGUOUS | Compare pay between departments.

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: 'Pay' unbound: base salary only or total comp; dept rankings differ.

### AMBIGUOUS | Who makes the most money?

- **schema**: `employees_multi_metric`
- **label**: AMBIGUOUS
- **why**: Same earnings fork as 'earns the most' (base vs stock vs total).

### UNDERSPECIFIED | Which employees earned over 8000 in March?

- **schema**: `employees_multi_metric`
- **label**: UNDERSPECIFIED
- **why**: No month/period column and salary is not per-period earnings; the temporal filter cannot be formed.

### UNDERSPECIFIED | Who is underpaid relative to the market rate?

- **schema**: `employees_multi_metric`
- **label**: UNDERSPECIFIED
- **why**: No market-rate reference data in schema.

### UNDERSPECIFIED | Who earns more than their manager?

- **schema**: `employees_multi_metric`
- **label**: UNDERSPECIFIED
- **why**: No manager column; reporting line absent.

### CLEAR | How many visits happened in March?

- **schema**: `vet_clinic_visits`
- **label**: CLEAR
- **why**: month column filter + COUNT; no competing reading.

### CLEAR | How many dogs weighed more than 15 kg?

- **schema**: `vet_clinic_visits`
- **label**: CLEAR
- **why**: species + weight_kg literal filters.

### CLEAR | Which pets require a follow-up visit?

- **schema**: `vet_clinic_visits`
- **label**: CLEAR
- **why**: followup_required is a dedicated flag; = 1 is the only reading.

### CLEAR | What is the average weight of cats compared to dogs?

- **schema**: `vet_clinic_visits`
- **label**: CLEAR
- **why**: AVG(weight_kg) per the two named species; no other metric involved.

### CLEAR | How many pets weigh more than the average?

- **schema**: `vet_clinic_visits`
- **label**: CLEAR
- **why**: Self-computable comparison (weight_kg > AVG over all visits); keyword rules naive flag 'compared to average' as needing clarification, it is well-defined.

### AMBIGUOUS | Which pets are the heaviest?

- **schema**: `vet_clinic_visits`
- **label**: AMBIGUOUS
- **why**: Scope fork: single globally heaviest pet vs heaviest per species; different SQL and different result shapes.

### AMBIGUOUS | Which pets cost us the most?

- **schema**: `vet_clinic_visits`
- **label**: AMBIGUOUS
- **why**: Cumulative SUM(visit_cost) per pet vs single costliest visit (MAX) - materially different.

### AMBIGUOUS | Which species had the costliest care overall?

- **schema**: `vet_clinic_visits`
- **label**: AMBIGUOUS
- **why**: Total spend per species vs average cost per visit per species; rankings can differ.

### AMBIGUOUS | Which months were the worst for the clinic?

- **schema**: `vet_clinic_visits`
- **label**: AMBIGUOUS
- **why**: Worst by revenue vs by visit volume; both computable from different columns.

### AMBIGUOUS | Which pets are the most valuable to the clinic?

- **schema**: `vet_clinic_visits`
- **label**: AMBIGUOUS
- **why**: Value could be total revenue (SUM visit_cost) or loyalty/repeat frequency (COUNT visits).

### UNDERSPECIFIED | List all vaccinations given in March.

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: No treatment/service/procedure column (only diagnosis); 'vaccinations' cannot be asserted from diagnosis text.

### UNDERSPECIFIED | Which pets are overweight for their size?

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: No reference weight/breed/baseline data; 'overweight' needs external criteria.

### UNDERSPECIFIED | Which pets need a dental check-up next?

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: No future scheduling or dental flags; followup_required is binary per recorded visit.

### UNDERSPECIFIED | What are the phone numbers of pets' owners?

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: Only owner_surname stored; no contact columns.

### UNDERSPECIFIED | Which pets are due for their annual booster shot?

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: No due-date/schedule data; months cannot be turned into due dates.

### UNDERSPECIFIED | Which pets have pet insurance?

- **schema**: `vet_clinic_visits`
- **label**: UNDERSPECIFIED
- **why**: No insurance column at all.

### CLEAR | How many rides lasted more than 30 minutes?

- **schema**: `bike_share_rides`
- **label**: CLEAR
- **why**: Literal filter on duration_minutes.

### CLEAR | Which start station is the most popular?

- **schema**: `bike_share_rides`
- **label**: CLEAR
- **why**: Station popularity = ride count in a rides table; single natural reading.

### CLEAR | What is the total fare collected in June?

- **schema**: `bike_share_rides`
- **label**: CLEAR
- **why**: SUM(fare_cents) with month filter.

### CLEAR | Which riders took more than 10 rides in September?

- **schema**: `bike_share_rides`
- **label**: CLEAR
- **why**: Per-rider COUNT with literal threshold and month; no distance/duration confusion.

### CLEAR | How many distinct bikes are in the data?

- **schema**: `bike_share_rides`
- **label**: CLEAR
- **why**: COUNT(DISTINCT bike_id).

### AMBIGUOUS | Which bike was used the most?

- **schema**: `bike_share_rides`
- **label**: AMBIGUOUS
- **why**: 'Most used' = most rides (COUNT) vs most minutes (SUM duration_minutes); naive keyword 'most' maps straight to a superlative and misses the fork.

### AMBIGUOUS | Which month was the busiest?

- **schema**: `bike_share_rides`
- **label**: AMBIGUOUS
- **why**: Busiest by ride count vs revenue vs total minutes; materially different queries.

### AMBIGUOUS | Which rider used the service the most?

- **schema**: `bike_share_rides`
- **label**: AMBIGUOUS
- **why**: Rides count vs total spend (fare_cents) vs total minutes.

### AMBIGUOUS | Which bike is our workhorse?

- **schema**: `bike_share_rides`
- **label**: AMBIGUOUS
- **why**: 'Workhorse' could be most rides, most time ridden, or most revenue; no single metric named.

### AMBIGUOUS | Which routes are the most popular?

- **schema**: `bike_share_rides`
- **label**: AMBIGUOUS
- **why**: Route popularity by number of rides vs by fares collected per (start,end) pair.

### UNDERSPECIFIED | How many riders rode more than 25 km?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No distance column (only duration and fare); numeric filter has nothing to compare.

### UNDERSPECIFIED | Which riders are students?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No rider profile attributes.

### UNDERSPECIFIED | Which bikes need preventive maintenance?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No usage/maintenance policy threshold; no service history.

### UNDERSPECIFIED | How many rides were cancelled?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No ride-status column (rows exist only for completed rides).

### UNDERSPECIFIED | What is the distance between Central Square and River Park?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No geo coordinates or inter-station distance data.

### UNDERSPECIFIED | Which riders have a monthly season pass?

- **schema**: `bike_share_rides`
- **label**: UNDERSPECIFIED
- **why**: No pass/subscription column.

### CLEAR | How many patients tested positive in March?

- **schema**: `cancer_screening_registry`
- **label**: CLEAR
- **why**: COUNT DISTINCT patient_code filtered by result='positive' and month; the counting unit (patients) resolves rows-vs-distinct.

### CLEAR | Which county had the most screenings?

- **schema**: `cancer_screening_registry`
- **label**: CLEAR
- **why**: Screening volume per county; 'most' = record count.

### CLEAR | What is the average age of patients screened in July?

- **schema**: `cancer_screening_registry`
- **label**: CLEAR
- **why**: AVG(age) with month filter.

### CLEAR | How many referrals are still pending?

- **schema**: `cancer_screening_registry`
- **label**: CLEAR
- **why**: referral_status='pending' is a literal state column.

### AMBIGUOUS | Which screening type has the most positives?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Highest positive COUNT vs highest positivity RATE per type - clinically different statements.

### AMBIGUOUS | Which county is doing the worst?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Lowest screening coverage (volume) vs highest positivity vs most pending referrals; policy-dependent.

### AMBIGUOUS | Which patients are high risk?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Age-based (age column) vs result-based (positive history) vs cost/referred-based risk; supported column forks.

### AMBIGUOUS | Who should we prioritize for follow-up outreach?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Priority could be positive results with pending referral, older patients, or recent high-cost rows; several column-supported policies.

### AMBIGUOUS | How effective is the screening programme in each county?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Effectiveness reads as detection yield (positivity rate), coverage (volume), or financial burden; no single metric named.

### AMBIGUOUS | Which month had the most concerning results?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: 'Concerning' = most positives vs highest positive rate vs most urgent referrals.

### AMBIGUOUS | Which age groups need the most screening?

- **schema**: `cancer_screening_registry`
- **label**: AMBIGUOUS
- **why**: Demand measured by screening volume per age band vs disease burden (positives) per age band.

### UNDERSPECIFIED | Which patients are overdue for their next screening?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No last-screening dates or interval policy; 'overdue' cannot be computed.

### UNDERSPECIFIED | Which patients died during follow-up?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No mortality or follow-up outcome data.

### UNDERSPECIFIED | How many women were screened?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No sex/gender column exists; naive keyword matching on 'screened' would miss it.

### UNDERSPECIFIED | Which patients have a family history of cancer?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No family-history table or column.

### UNDERSPECIFIED | Which results turned out to be false positives?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No ground-truth/verification data to flag false positives; result labels are surface readings only.

### UNDERSPECIFIED | What is the 5-year survival rate by county?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No outcome/follow-up data at all.

### UNDERSPECIFIED | What are patients' contact details for outreach?

- **schema**: `cancer_screening_registry`
- **label**: UNDERSPECIFIED
- **why**: No phone/address/email columns in the registry.

