"""
Generate the synthetic supplier proposals used by the demo and the test suite.

The four proposals are fixtures, not filler. Each one is deliberately shaped to
exercise a different part of the pipeline:

* **Apex Systems** — complete and specific on every criterion. The expected
  benchmark setter, and the fixture the retrieval tests assert against.
* **BrightPath Tech** — cheapest and technically sound, but holds no security
  certification. Exercises a low score backed by real evidence.
* **NexaWorks** — strong security, pricing explicitly deferred. Exercises a
  partially specified commercial criterion.
* **Orbit Digital** — strong references, but silent on implementation and
  pricing. Exercises the "not specified" path, where absence must not be scored
  as proven non-compliance.

Every line is drawn individually with `drawString` rather than flowed through a
paragraph engine, so the extracted line numbers are an exact, stable function of
this file. That is what lets the search tests assert on specific page and line
citations. `invariant=1` keeps the PDF bytes reproducible across runs.

Usage::

    python generate_sample_pdfs.py            # write any missing proposals
    python generate_sample_pdfs.py --force    # rewrite all of them
    python generate_sample_pdfs.py --out dir  # write somewhere else
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 50
FONT_NAME = "Helvetica"
FONT_SIZE = 8
LEADING = 13

#: filename stem -> list of pages, each page a list of lines.
#: Line N of page P is exactly the Nth string below, which is what the retrieval
#: tests cite. Reordering these lines will change those citations.
PROPOSALS: Dict[str, List[List[str]]] = {
    "Apex_Systems": [
        [
            "Apex Systems - RFP Response: Core Platform Modernisation",            # 1
            "Submission date: 2026-01-12. Contact: bids@apexsystems.example",      # 2
            "1. Technical Capability",                                             # 3
            "Apex proposes a cloud-native architecture on Kubernetes across three availability zones.",   # 4
            "All integrations are delivered as versioned REST APIs with an event bus for batch loads.",   # 5
            "The platform scales horizontally to 12,000 concurrent users with per-service autoscaling.",  # 6
            "Our disaster recovery design provides a 15-minute RPO and a 1-hour RTO with auto failover.",  # 7
            "Disaster recovery is tested quarterly and results are published to the client steering group.",  # 8
            "2. Implementation Plan",                                              # 9
            "Delivery runs over 16 weeks across four milestones, with a named lead and six engineers.",   # 10
            "Milestone gates cover discovery, build, integration testing and go-live readiness review.",  # 11
            "A documented rollback plan restores the previous release within 30 minutes at any gate.",    # 12
            "3. Commercial Value",                                                 # 13
            "Fixed implementation fee: $240,000 covering design, build, migration and go-live support.",  # 14
            "Annual support cost: $36,000 per year for three years, with no indexation applied.",         # 15
            "Total three-year cost of ownership: $348,000. All assumptions are listed in Appendix B.",     # 16
        ],
        [
            "4. Security and Compliance",                                          # 1
            "Apex holds ISO 27001 certification, renewed March 2026, and SOC 2 Type II attestation.",     # 2
            "Data is encrypted in transit with TLS 1.3 and at rest with AES-256 customer-managed keys.",  # 3
            "Immutable audit logs capture every privileged action and are retained for seven years.",     # 4
            "Annual penetration testing is performed by an external CREST-accredited assessor.",          # 5
            "5. Support and Experience",                                           # 6
            "Support is 24x7 with a 1-hour response for critical incidents and a 99.9% availability SLA.",  # 7
            "Apex has delivered 14 comparable modernisation programmes in regulated sectors since 2019.",   # 8
            "Three client references are available, including two public-sector organisations.",           # 9
        ],
    ],
    "BrightPath_Tech": [
        [
            "BrightPath Tech - RFP Response: Platform Delivery Proposal",          # 1
            "Submission date: 2026-01-09. Contact: proposals@brightpath.example",  # 2
            "1. Technical Capability",                                             # 3
            "BrightPath delivers a containerised microservices platform hosted on managed infrastructure.",  # 4
            "Integration is provided through REST endpoints and nightly scheduled file transfers.",          # 5
            "The design supports 4,000 concurrent users; load beyond that requires a commercial review.",    # 6
            "Nightly backup and restore procedures are documented, with a 24-hour recovery objective.",      # 7
            "2. Implementation Plan",                                              # 8
            "Delivery is phased over 12 weeks with three milestones and a shared delivery manager.",         # 9
            "Testing covers unit, integration and user acceptance stages before each release gate.",         # 10
            "Rollback is handled by redeploying the prior container image, typically within two hours.",     # 11
            "3. Commercial Value",                                                 # 12
            "Fixed implementation fee: $155,000, the lowest total cost offered in this submission.",         # 13
            "Annual support cost: $22,000 per year, fixed for the first two years of operation.",            # 14
            "Total three-year cost of ownership: $221,000, itemised by workstream in Appendix A.",           # 15
        ],
        [
            "4. Security and Compliance",                                          # 1
            "BrightPath is not currently ISO 27001 certified; certification is targeted for late 2027.",     # 2
            "Data is encrypted in transit using TLS 1.2 and at rest using provider-managed encryption.",     # 3
            "Application logs are retained for 90 days; privileged action audit logging is not itemised.",   # 4
            "5. Support and Experience",                                           # 5
            "Support operates business hours with a 4-hour response target for critical incidents.",         # 6
            "BrightPath has completed five comparable platform deliveries, primarily in retail.",            # 7
            "Two client references are available on request.",                     # 8
        ],
    ],
    "NexaWorks": [
        [
            "NexaWorks - RFP Response: Secure Platform Programme",                 # 1
            "Submission date: 2026-01-15. Contact: tenders@nexaworks.example",     # 2
            "1. Technical Capability",                                             # 3
            "NexaWorks proposes a hybrid deployment with a private cloud control plane and public edge.",    # 4
            "Integrations use a documented REST API gateway with mutual TLS between all services.",          # 5
            "Capacity planning targets 8,000 concurrent users, validated by load testing evidence.",         # 6
            "A disaster recovery runbook exists with a 4-hour RTO; failover is operator-initiated.",         # 7
            "2. Implementation Plan",                                              # 8
            "Delivery spans 20 weeks across five milestones, with a governance board meeting fortnightly.",  # 9
            "Staffing is confirmed for discovery and build; later phases are resourced on demand.",          # 10
            "Rollback procedures are described at programme level but not per milestone.",                   # 11
            "3. Commercial Value",                                                 # 12
            "Pricing is indicative only at this stage and will be confirmed after discovery.",               # 13
            "An implementation range of $180,000 to $260,000 is offered, subject to final scope.",           # 14
            "Annual support cost has not been fixed and will be proposed in the contract schedule.",         # 15
        ],
        [
            "4. Security and Compliance",                                          # 1
            "NexaWorks holds ISO 27001 and ISO 27017 certification and is audited annually.",                # 2
            "All data is encrypted with AES-256 at rest and TLS 1.3 in transit, with HSM-backed keys.",      # 3
            "Immutable audit logs are retained for ten years and are exportable to client SIEM tooling.",    # 4
            "The platform is assessed under a formal information security management system.",              # 5
            "Quarterly penetration tests and continuous vulnerability scanning are included.",              # 6
            "5. Support and Experience",                                           # 7
            "Support is 24x7 for security incidents and business hours for standard service requests.",      # 8
            "NexaWorks has delivered nine programmes in financial services and healthcare.",                 # 9
        ],
    ],
    "Orbit_Digital": [
        [
            "Orbit Digital - RFP Response: Partnership Submission",                # 1
            "Submission date: 2026-01-09. Contact: hello@orbitdigital.example",    # 2
            "1. Technical Capability",                                             # 3
            "Orbit Digital provides a modular platform built on established open-source components.",        # 4
            "Integration options include REST and webhook delivery, scoped during onboarding.",              # 5
            "Business continuity arrangements are maintained in line with our internal policy.",             # 6
            "2. Our Approach",                                                     # 7
            "We work in two-week iterations with continuous stakeholder engagement throughout delivery.",    # 8
            "Detailed milestones, staffing profiles and dependencies will be agreed after contract award.",  # 9
            "3. Experience and References",                                        # 10
            "Orbit Digital has supported 23 clients over 11 years with an average tenure of six years.",     # 11
            "Support is 24x7 with a 30-minute response for critical incidents and a 99.95% uptime SLA.",     # 12
            "A dedicated customer success manager is assigned to every engagement at no extra cost.",        # 13
            "Six referenceable clients are listed in Appendix C, four in the same sector as this RFP.",      # 14
            "Client satisfaction averaged 4.8 out of 5 across the last three annual surveys.",               # 15
        ],
        [
            "4. Security",                                                         # 1
            "Orbit Digital follows documented secure development practices and reviews access quarterly.",   # 2
            "Formal certification status and encryption standards are available on request.",               # 3
            "5. Commercial",                                                       # 4
            "A commercial proposal will be issued separately once requirements are finalised.",             # 5
        ],
    ],
}


def _render(path: Path, pages: List[List[str]]) -> None:
    """Draw one proposal, one line per list entry, deterministically."""
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    pdf.setTitle(path.stem.replace("_", " "))
    pdf.setAuthor("Synthetic RFP fixture")
    pdf.setSubject("Synthetic supplier proposal for testing")

    usable_width = PAGE_WIDTH - 2 * MARGIN
    for page_index, lines in enumerate(pages):
        pdf.setFont(FONT_NAME, FONT_SIZE)
        y = PAGE_HEIGHT - MARGIN
        for line in lines:
            if pdf.stringWidth(line, FONT_NAME, FONT_SIZE) > usable_width:
                print(f"  warning: line exceeds the text frame and may wrap: {line[:60]}…",
                      file=sys.stderr)
            pdf.drawString(MARGIN, y, line)
            y -= LEADING
        pdf.setFont(FONT_NAME, 7)
        pdf.drawString(MARGIN, MARGIN / 2, f"{path.stem.replace('_', ' ')} - page {page_index + 1}")
        pdf.showPage()
    pdf.save()


def generate(output_dir: Path, force: bool = False) -> List[Path]:
    """Write the proposals, skipping existing files unless ``force`` is set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for stem, pages in PROPOSALS.items():
        path = output_dir / f"{stem}.pdf"
        if path.exists() and not force:
            print(f"skip   {path} (already exists; use --force to rewrite)")
            continue
        _render(path, pages)
        line_count = sum(len(page) for page in pages)
        print(f"write  {path} ({len(pages)} pages, {line_count} lines)")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic supplier proposal PDFs.")
    parser.add_argument("--out", default=str(Path(__file__).parent / "sample_pdfs"),
                        help="Output directory (default: ./sample_pdfs)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    generate(Path(args.out), force=args.force)
    print(f"\n{len(PROPOSALS)} proposal(s) available in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
