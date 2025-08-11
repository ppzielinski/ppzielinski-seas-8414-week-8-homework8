# Testing Guide

This document outlines manual test cases to verify both **prediction** and **attribution** features.  
Each test lists **all available UI options** as they appear in the Streamlit sidebar.

---

## 1) Benign URL Test
**Purpose:** Ensure a clean URL is classified as *Benign* and no attribution is performed.

**Input (set ALL options exactly as follows):**
- **URL Length:** Short
- **SSL Status:** Trusted
- **Sub-domain:** None
- **Has Prefix/Suffix:** Unchecked
- **Uses IP Address:** Unchecked
- **Is Shortened:** Unchecked
- **Has '@':** Unchecked
- **Is Abnormal:** Unchecked
- **Has Political Keyword:** Unchecked

**Expected Output:**
- **Verdict:** Benign
- **Threat Actor Profile:** N/A

---

## 2) Malicious — State-Sponsored Profile
**Purpose:** Validate attribution for a high-sophistication, stealthy attack.

**Input (set ALL options exactly as follows):**
- **URL Length:** Short
- **SSL Status:** Trusted
- **Sub-domain:** One
- **Has Prefix/Suffix:** Checked
- **Uses IP Address:** Unchecked
- **Is Shortened:** Unchecked
- **Has '@':** Unchecked
- **Is Abnormal:** Checked
- **Has Political Keyword:** Unchecked

**Expected Output:**
- **Verdict:** Malicious
- **Threat Actor Profile:** State-Sponsored

---

## 3) Malicious — Organized Cybercrime Profile
**Purpose:** Validate attribution for noisy, high-volume attack patterns.

**Input (set ALL options exactly as follows):**
- **URL Length:** Long
- **SSL Status:** Suspicious
- **Sub-domain:** Many
- **Has Prefix/Suffix:** Checked
- **Uses IP Address:** Checked
- **Is Shortened:** Checked
- **Has '@':** Checked
- **Is Abnormal:** Checked
- **Has Political Keyword:** Unchecked

**Expected Output:**
- **Verdict:** Malicious
- **Threat Actor Profile:** Organized Cybercrime

---

## 4) Malicious — Hacktivist Profile
**Purpose:** Validate attribution for politically motivated campaigns.

**Input (set ALL options exactly as follows):**
- **URL Length:** Normal
- **SSL Status:** None
- **Sub-domain:** One
- **Has Prefix/Suffix:** Unchecked
- **Uses IP Address:** Unchecked
- **Is Shortened:** Unchecked
- **Has '@':** Unchecked
- **Is Abnormal:** Unchecked
- **Has Political Keyword:** Checked

**Expected Output:**
- **Verdict:** Malicious
- **Threat Actor Profile:** Hacktivist

---

## Notes
- Run tests through the Streamlit UI sidebar controls.
- Check both the **Prediction** and **Threat Attribution** tabs for correct results.