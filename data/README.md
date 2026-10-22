# Demonstration data

This directory contains fixtures for local demos, model training, evaluation, and automated tests:

| Directory | Contents |
| --- | --- |
| `flows/` | Eight declarative support workflows |
| `intents/` | Training, evaluation, and test utterances with intent labels |
| `golden/` | Conversation scripts with expected intents, text fragments, state, and outcomes |
| `kb/` | An example FAQ corpus for extractive knowledge retrieval |

The sample order and appointment backend lives in `src/conversaflow/agents/tools.py`. Its identifiers, dates, amounts, locations, and outcomes are deterministic demonstration records. Example contact strings and street addresses in tests or conversations are fixtures, not a connection to a customer database.

## Fictional company policies

The knowledge documents in `kb/faq.jsonl` describe a fictional company and use `help.example.com` source URLs. Their statements about certifications, GDPR compliance, encryption, retention, service availability, enterprise SSO, mobile applications, pricing, and integrations are sample retrieval content.

**Those statements are not certifications, compliance assurances, product commitments, or implemented features of ConversaFlow AI.** The runtime returns passages from its corpus; it does not independently verify the truth of a source document.

Replace the FAQ with reviewed domain documents before deployment. Train on appropriate domain examples and evaluate against independently labeled inputs. Do not add live credentials, identifiable customer transcripts, or proprietary business records to this public fixture directory.
