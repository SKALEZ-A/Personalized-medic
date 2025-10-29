"""
Comprehensive Genomic Algorithms for AI Personalized Medicine Platform
"""

import re
import math
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter

class GenomicAlgorithms:
    """Advanced genomic analysis algorithms"""

    def __init__(self):
        self.reference_genomes = self._load_reference_genomes()
        self.variant_databases = self._load_variant_databases()
        self.gene_annotations = self._load_gene_annotations()

    def _load_reference_genomes(self) -> Dict[str, str]:
        """Load reference genome sequences (simplified)"""
        # In real implementation, would load from FASTA files
        return {
            "GRCh38": {
                "chr1": "ATCG" * 1000,  # Simplified
                "chr2": "GCTA" * 1000,
                # ... more chromosomes
            }
        }

    def _load_variant_databases(self) -> Dict[str, Dict[str, Any]]:
        """Load variant databases (ClinVar, dbSNP, etc.)"""
        # Simplified variant database
        return {
            "rs123456": {
                "clinical_significance": "pathogenic",
                "disease": "cystic_fibrosis",
                "frequency": 0.001
            },
            "rs789012": {
                "clinical_significance": "benign",
                "disease": None,
                "frequency": 0.45
            }
        }

    def _load_gene_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load gene annotations and functions"""
        return {
            "CFTR": {
                "chromosome": "chr7",
                "start": 117149127,
                "end": 117311719,
                "function": "chloride channel",
                "associated_diseases": ["cystic_fibrosis"],
                "pharmacogenomics": ["ivacaftor"]
            },
            "TPMT": {
                "chromosome": "chr6",
                "start": 18126596,
                "end": 18157519,
                "function": "thiopurine metabolism",
                "associated_diseases": ["thiopurine_toxicity"],
                "pharmacogenomics": ["azathioprine", "mercaptopurine"]
            }
        }

    def analyze_genome_sequence(self, genome_sequence: str) -> Dict[str, Any]:
        """Comprehensive genome sequence analysis"""
        results = {
            "sequence_quality": self._assess_sequence_quality(genome_sequence),
            "gc_content": self._calculate_gc_content(genome_sequence),
            "nucleotide_distribution": self._analyze_nucleotide_distribution(genome_sequence),
            "repetitive_elements": self._identify_repetitive_elements(genome_sequence),
            "potential_variants": self._identify_potential_variants(genome_sequence),
            "gene_content": self._analyze_gene_content(genome_sequence)
        }

        return results

    def _assess_sequence_quality(self, sequence: str) -> Dict[str, Any]:
        """Assess overall sequence quality"""
        length = len(sequence)
        n_count = sequence.upper().count('N')
        quality_score = 1 - (n_count / length)

        # Phred-like quality score
        phred_score = -10 * math.log10(max(1 - quality_score, 0.001))

        return {
            "length": length,
            "n_bases": n_count,
            "quality_score": quality_score,
            "phred_score": phred_score,
            "quality_category": "high" if quality_score > 0.9 else "medium" if quality_score > 0.7 else "low"
        }

    def _calculate_gc_content(self, sequence: str) -> Dict[str, Any]:
        """Calculate GC content and analyze its distribution"""
        seq = sequence.upper()
        gc_count = seq.count('G') + seq.count('C')
        at_count = seq.count('A') + seq.count('T')
        total = gc_count + at_count

        if total == 0:
            return {"gc_content": 0, "gc_skew": 0, "distribution": "uniform"}

        gc_content = gc_count / total

        # GC skew
        gc_skew = (seq.count('G') - seq.count('C')) / (seq.count('G') + seq.count('C')) if (seq.count('G') + seq.count('C')) > 0 else 0

        # Local GC content variation
        window_size = 100
        gc_windows = []

        for i in range(0, len(seq) - window_size + 1, window_size):
            window = seq[i:i + window_size]
            window_gc = (window.count('G') + window.count('C')) / window_size
            gc_windows.append(window_gc)

        gc_variation = statistics.stdev(gc_windows) if len(gc_windows) > 1 else 0

        return {
            "gc_content": gc_content,
            "gc_skew": gc_skew,
            "gc_variation": gc_variation,
            "distribution": "variable" if gc_variation > 0.1 else "uniform"
        }

    def _analyze_nucleotide_distribution(self, sequence: str) -> Dict[str, Any]:
        """Analyze nucleotide distribution and patterns"""
        seq = sequence.upper()
        nucleotides = ['A', 'T', 'G', 'C', 'N']

        counts = {nuc: seq.count(nuc) for nuc in nucleotides}
        total = sum(counts.values())

        frequencies = {nuc: count / total for nuc, count in counts.items()}

        # Dinucleotide frequencies
        dinucleotides = {}
        for i in range(len(seq) - 1):
            di = seq[i:i + 2]
            dinucleotides[di] = dinucleotides.get(di, 0) + 1

        total_di = sum(dinucleotides.values())
        di_frequencies = {di: count / total_di for di, count in dinucleotides.items()}

        # CpG islands (simplified)
        cpg_sites = seq.count('CG')
        expected_cpg = (seq.count('C') * seq.count('G')) / len(seq)
        cpg_oe_ratio = cpg_sites / expected_cpg if expected_cpg > 0 else 0

        return {
            "nucleotide_counts": counts,
            "nucleotide_frequencies": frequencies,
            "dinucleotide_frequencies": di_frequencies,
            "cpg_sites": cpg_sites,
            "cpg_oe_ratio": cpg_oe_ratio,
            "cpg_islands": cpg_oe_ratio > 1.5
        }

    def _identify_repetitive_elements(self, sequence: str) -> Dict[str, Any]:
        """Identify repetitive elements in the sequence"""
        seq = sequence.upper()

        # Simple repeat identification
        repeats = {}
        min_repeat_length = 6

        for length in range(min_repeat_length, min_repeat_length + 5):
            for i in range(len(seq) - length + 1):
                pattern = seq[i:i + length]
                count = seq.count(pattern)
                if count > 1:
                    repeats[pattern] = count

        # Tandem repeats
        tandem_repeats = []
        for pattern, count in repeats.items():
            if count > 2:  # At least 3 occurrences
                tandem_repeats.append({
                    "pattern": pattern,
                    "length": len(pattern),
                    "count": count,
                    "total_bases": len(pattern) * count
                })

        # Microsatellites (simple repeats)
        microsatellites = []
        for pattern in ["A", "T", "G", "C", "AT", "GC", "AG", "CT"]:
            extended_pattern = pattern * 6  # At least 6 repeats
            positions = [m.start() for m in re.finditer(extended_pattern, seq)]
            if positions:
                microsatellites.append({
                    "pattern": pattern,
                    "repeat_unit": len(pattern),
                    "locations": positions,
                    "count": len(positions)
                })

        return {
            "total_repeats": len(repeats),
            "tandem_repeats": tandem_repeats[:10],  # Top 10
            "microsatellites": microsatellites,
            "repeat_content": sum(len(p) * c for p, c in repeats.items()) / len(seq)
        }

    def _identify_potential_variants(self, sequence: str) -> List[Dict[str, Any]]:
        """Identify potential genetic variants"""
        variants = []

        # Simple SNP identification (compared to reference)
        reference = self.reference_genomes["GRCh38"].get("chr1", "ATCG" * 1000)

        min_length = min(len(sequence), len(reference))
        sequence = sequence[:min_length]
        reference = reference[:min_length]

        for i in range(min_length):
            if sequence[i] != reference[i] and sequence[i] != 'N' and reference[i] != 'N':
                variants.append({
                    "position": i + 1,
                    "reference": reference[i],
                    "alternate": sequence[i],
                    "type": "SNP",
                    "quality": 30,  # Default quality
                    "annotations": self._annotate_variant(reference[i], sequence[i], i + 1)
                })

        # Indels (simplified)
        # In real implementation, would use alignment algorithms

        return variants[:100]  # Limit for performance

    def _annotate_variant(self, ref: str, alt: str, position: int) -> Dict[str, Any]:
        """Annotate variant with functional information"""
        annotations = {
            "functional_impact": "unknown",
            "conservation_score": 0.5,
            "population_frequency": 0.01,
            "clinical_significance": "unknown"
        }

        # Simple functional prediction
        if ref in ['A', 'T'] and alt in ['G', 'C'] or ref in ['G', 'C'] and alt in ['A', 'T']:
            annotations["functional_impact"] = "transition"
        else:
            annotations["functional_impact"] = "transversion"

        # Conservation based on GC content
        if (ref in ['G', 'C'] and alt in ['A', 'T']) or (ref in ['A', 'T'] and alt in ['G', 'C']):
            annotations["conservation_score"] = 0.8  # Higher impact

        return annotations

    def _analyze_gene_content(self, sequence: str) -> Dict[str, Any]:
        """Analyze gene content and coding potential"""
        # Simplified gene prediction
        seq = sequence.upper()

        # Codon analysis
        codons = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if 'N' not in codon:
                codons.append(codon)

        # Open reading frames (simplified)
        orfs = []
        start_codon = "ATG"
        stop_codons = ["TAA", "TAG", "TGA"]

        for frame in range(3):
            frame_sequence = seq[frame:]
            orf_start = None

            for i in range(0, len(frame_sequence) - 2, 3):
                codon = frame_sequence[i:i + 3]

                if codon == start_codon and orf_start is None:
                    orf_start = i + frame
                elif codon in stop_codons and orf_start is not None:
                    orf_length = (i + frame) - orf_start
                    if orf_length > 30:  # Minimum ORF length
                        orfs.append({
                            "start": orf_start,
                            "end": i + frame + 2,
                            "length": orf_length,
                            "frame": frame
                        })
                    orf_start = None

        # GC content in coding regions (simplified)
        coding_gc = 0.45  # Default
        if orfs:
            coding_regions = []
            for orf in orfs:
                coding_regions.extend(range(orf["start"], orf["end"] + 1))
            if coding_regions:
                coding_bases = [seq[i] for i in coding_regions if i < len(seq)]
                gc_bases = sum(1 for base in coding_bases if base in ['G', 'C'])
                coding_gc = gc_bases / len(coding_bases) if coding_bases else 0.45

        return {
            "total_codons": len(codons),
            "open_reading_frames": len(orfs),
            "orf_details": orfs[:5],  # Top 5 ORFs
            "coding_gc_content": coding_gc,
            "coding_potential": "high" if len(orfs) > 0 else "low"
        }

    def perform_variant_calling(self, genome_sequence: str) -> Dict[str, Any]:
        """Comprehensive variant calling pipeline"""
        # Quality filtering
        quality_filtered = self._quality_filter_sequence(genome_sequence)

        # Alignment (simplified)
        alignment = self._align_to_reference(quality_filtered)

        # Variant identification
        variants = self._call_variants(alignment)

        # Variant filtering and annotation
        filtered_variants = self._filter_variants(variants)
        annotated_variants = self._annotate_variants(filtered_variants)

        return {
            "total_variants": len(variants),
            "filtered_variants": len(filtered_variants),
            "annotated_variants": annotated_variants,
            "variant_types": Counter(v["type"] for v in annotated_variants),
            "quality_metrics": {
                "average_quality": statistics.mean(v.get("quality", 0) for v in annotated_variants),
                "pass_filter_rate": len(filtered_variants) / len(variants) if variants else 0
            }
        }

    def _quality_filter_sequence(self, sequence: str) -> str:
        """Apply quality filtering to sequence"""
        # Simple quality filtering - remove low quality bases
        filtered = []
        for base in sequence:
            if base != 'N':  # Remove unknown bases
                filtered.append(base)
        return ''.join(filtered)

    def _align_to_reference(self, sequence: str) -> Dict[str, Any]:
        """Align sequence to reference genome (simplified)"""
        reference = self.reference_genomes["GRCh38"].get("chr1", "ATCG" * 1000)

        # Simple alignment - in reality would use BWA, Bowtie, etc.
        alignment_score = sum(1 for a, b in zip(sequence, reference) if a == b)

        return {
            "reference": reference,
            "query": sequence,
            "alignment_score": alignment_score,
            "identity": alignment_score / len(sequence),
            "cigar": f"{len(sequence)}M"  # Simplified CIGAR string
        }

    def _call_variants(self, alignment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call variants from alignment"""
        variants = []
        reference = alignment["reference"]
        query = alignment["query"]

        for i, (ref_base, query_base) in enumerate(zip(reference, query)):
            if ref_base != query_base and ref_base != 'N' and query_base != 'N':
                variants.append({
                    "chromosome": "chr1",
                    "position": i + 1,
                    "reference": ref_base,
                    "alternate": query_base,
                    "type": "SNP",
                    "quality": 30,
                    "depth": 30,
                    "allele_frequency": 1.0
                })

        return variants

    def _filter_variants(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter variants based on quality criteria"""
        filtered = []

        for variant in variants:
            # Quality filters
            if (variant.get("quality", 0) >= 20 and
                variant.get("depth", 0) >= 10 and
                variant.get("allele_frequency", 0) >= 0.2):
                filtered.append(variant)

        return filtered

    def _annotate_variants(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate variants with functional and clinical information"""
        annotated = []

        for variant in variants:
            # Create variant ID
            variant_id = f"{variant['chromosome']}:{variant['position']}_{variant['reference']}>{variant['alternate']}"

            # Functional annotation
            functional_annotation = self._get_functional_annotation(variant)

            # Clinical annotation
            clinical_annotation = self._get_clinical_annotation(variant_id)

            # Population frequency
            population_freq = self._get_population_frequency(variant_id)

            annotated_variant = variant.copy()
            annotated_variant.update({
                "variant_id": variant_id,
                "functional_annotation": functional_annotation,
                "clinical_annotation": clinical_annotation,
                "population_frequency": population_freq,
                "conservation_score": self._calculate_conservation_score(variant)
            })

            annotated.append(annotated_variant)

        return annotated

    def _get_functional_annotation(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional annotation for variant"""
        # Simplified functional prediction
        functional_impact = "modifier"  # Default

        # Check if in coding region (simplified)
        if variant.get("position", 0) % 3 == 0:  # Rough coding region check
            functional_impact = "missense_variant"

        return {
            "impact": functional_impact,
            "gene": "GENE1",  # Would be determined by position
            "feature_type": "transcript",
            "transcript_biotype": "protein_coding"
        }

    def _get_clinical_annotation(self, variant_id: str) -> Dict[str, Any]:
        """Get clinical annotation from databases"""
        # Check variant databases
        if variant_id in self.variant_databases:
            db_entry = self.variant_databases[variant_id]
            return {
                "clinical_significance": db_entry["clinical_significance"],
                "review_status": "reviewed_by_expert_panel",
                "disease_association": db_entry["disease"]
            }

        return {
            "clinical_significance": "unknown",
            "review_status": "not_reviewed",
            "disease_association": None
        }

    def _get_population_frequency(self, variant_id: str) -> Dict[str, Any]:
        """Get population frequency data"""
        # Simplified - would query 1000 Genomes, gnomAD, etc.
        if variant_id in self.variant_databases:
            return {
                "global_frequency": self.variant_databases[variant_id]["frequency"],
                "african_frequency": 0.02,
                "european_frequency": 0.01,
                "asian_frequency": 0.005
            }

        return {
            "global_frequency": 0.001,  # Default rare variant
            "african_frequency": 0.002,
            "european_frequency": 0.001,
            "asian_frequency": 0.0005
        }

    def _calculate_conservation_score(self, variant: Dict[str, Any]) -> float:
        """Calculate conservation score"""
        # Simplified conservation score based on position and type
        base_score = 0.5

        # Higher conservation for coding regions
        if variant.get("position", 0) % 3 == 0:
            base_score += 0.3

        # Higher conservation for transitions
        ref, alt = variant.get("reference", ""), variant.get("alternate", "")
        if ((ref in ['A', 'G'] and alt in ['A', 'G']) or
            (ref in ['T', 'C'] and alt in ['T', 'C'])):
            base_score += 0.1

        return min(base_score, 1.0)

    def analyze_pharmacogenomics(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pharmacogenomic implications of variants"""
        pharmacogenomic_findings = {
            "drug_metabolism": {},
            "drug_transport": {},
            "drug_targets": {},
            "adverse_reactions": [],
            "dosage_recommendations": {}
        }

        # Check for pharmacogenomic genes
        pgx_variants = []
        for variant in variants:
            if self._is_pharmacogenomic_variant(variant):
                pgx_variants.append(variant)

        # Analyze drug metabolism genes
        metabolism_analysis = self._analyze_drug_metabolism(pgx_variants)
        pharmacogenomic_findings["drug_metabolism"] = metabolism_analysis

        # Analyze drug transport
        transport_analysis = self._analyze_drug_transport(pgx_variants)
        pharmacogenomic_findings["drug_transport"] = transport_analysis

        # Generate dosage recommendations
        dosage_rec = self._generate_dosage_recommendations(metabolism_analysis)
        pharmacogenomic_findings["dosage_recommendations"] = dosage_rec

        return pharmacogenomic_findings

    def _is_pharmacogenomic_variant(self, variant: Dict[str, Any]) -> bool:
        """Check if variant has pharmacogenomic implications"""
        # Check if variant is in known pharmacogenomic genes
        variant_pos = variant.get("position", 0)

        for gene, info in self.gene_annotations.items():
            if (info.get("start", 0) <= variant_pos <= info.get("end", 0) and
                "pharmacogenomics" in info):
                return True

        return False

    def _analyze_drug_metabolism(self, pgx_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze drug metabolism capacity"""
        metabolism_status = {}

        # CYP2D6 analysis
        cyp2d6_variants = [v for v in pgx_variants if "CYP2D6" in str(v)]
        if cyp2d6_variants:
            metabolism_status["CYP2D6"] = self._classify_cyp2d6_metabolizer(cyp2d6_variants)
        else:
            metabolism_status["CYP2D6"] = "normal_metabolizer"

        # CYP2C19 analysis
        cyp2c19_variants = [v for v in pgx_variants if "CYP2C19" in str(v)]
        if cyp2c19_variants:
            metabolism_status["CYP2C19"] = self._classify_cyp2c19_metabolizer(cyp2c19_variants)
        else:
            metabolism_status["CYP2C19"] = "normal_metabolizer"

        return metabolism_status

    def _classify_cyp2d6_metabolizer(self, variants: List[Dict[str, Any]]) -> str:
        """Classify CYP2D6 metabolizer status"""
        # Simplified classification based on variant count
        variant_count = len(variants)

        if variant_count >= 3:
            return "poor_metabolizer"
        elif variant_count >= 2:
            return "intermediate_metabolizer"
        elif variant_count >= 1:
            return "ultrarapid_metabolizer"
        else:
            return "normal_metabolizer"

    def _classify_cyp2c19_metabolizer(self, variants: List[Dict[str, Any]]) -> str:
        """Classify CYP2C19 metabolizer status"""
        # Similar simplified classification
        variant_count = len(variants)

        if variant_count >= 2:
            return "poor_metabolizer"
        elif variant_count >= 1:
            return "intermediate_metabolizer"
        else:
            return "normal_metabolizer"

    def _analyze_drug_transport(self, pgx_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze drug transport capacity"""
        transport_status = {}

        # SLCO1B1 analysis (statin transport)
        slco1b1_variants = [v for v in pgx_variants if "SLCO1B1" in str(v)]
        if slco1b1_variants:
            transport_status["SLCO1B1"] = "reduced_transport"
        else:
            transport_status["SLCO1B1"] = "normal_transport"

        return transport_status

    def _generate_dosage_recommendations(self, metabolism_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate drug dosage recommendations"""
        recommendations = {}

        # Common drugs affected by CYP2D6
        if metabolism_analysis.get("CYP2D6") == "poor_metabolizer":
            recommendations.update({
                "codeine": "avoid_or_reduce_dose",
                "tamoxifen": "consider_alternative",
                "tramadol": "reduce_dose",
                "risperidone": "reduce_dose"
            })

        # Common drugs affected by CYP2C19
        if metabolism_analysis.get("CYP2C19") == "poor_metabolizer":
            recommendations.update({
                "clopidogrel": "consider_alternative",
                "omeprazole": "adjust_dose",
                "citalopram": "reduce_dose"
            })

        return recommendations

    def calculate_disease_risk_scores(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate disease risk scores from variants"""
        risk_scores = {}

        # Cardiovascular disease risk
        cvd_risk = self._calculate_cvd_risk(variants)
        risk_scores["cardiovascular_disease"] = cvd_risk

        # Cancer risk
        cancer_risk = self._calculate_cancer_risk(variants)
        risk_scores["cancer"] = cancer_risk

        # Diabetes risk
        diabetes_risk = self._calculate_diabetes_risk(variants)
        risk_scores["diabetes"] = diabetes_risk

        # Alzheimer's risk
        alzheimer_risk = self._calculate_alzheimer_risk(variants)
        risk_scores["alzheimer"] = alzheimer_risk

        return risk_scores

    def _calculate_cvd_risk(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cardiovascular disease risk"""
        base_risk = 0.1  # Base population risk

        # Risk-increasing variants
        risk_variants = [
            "rs1333049",  # 9p21 locus
            "rs10757274",  # 9p21 locus
            "rs2383206"   # PCSK9 locus
        ]

        risk_count = sum(1 for v in variants if v.get("variant_id", "").split(":")[0] in risk_variants)
        adjusted_risk = base_risk * (1 + risk_count * 0.5)

        return {
            "risk_score": min(adjusted_risk, 1.0),
            "risk_category": "high" if adjusted_risk > 0.3 else "moderate" if adjusted_risk > 0.15 else "low",
            "contributing_variants": risk_count,
            "lifestyle_modifier": 0.8  # Can be modified by lifestyle factors
        }

    def _calculate_cancer_risk(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cancer risk"""
        base_risk = 0.05  # Base population risk

        # High-risk cancer variants
        high_risk_variants = [
            "rs1801133",  # MTHFR
            "rs1801320",  # BRCA2
            "rs28897743"  # PALB2
        ]

        high_risk_count = sum(1 for v in variants if v.get("variant_id", "").split(":")[0] in high_risk_variants)
        adjusted_risk = base_risk * (1 + high_risk_count * 2.0)

        return {
            "risk_score": min(adjusted_risk, 1.0),
            "risk_category": "high" if adjusted_risk > 0.2 else "moderate" if adjusted_risk > 0.1 else "low",
            "contributing_variants": high_risk_count,
            "screening_recommendations": ["annual_mammogram"] if adjusted_risk > 0.15 else []
        }

    def _calculate_diabetes_risk(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diabetes risk"""
        base_risk = 0.08  # Base population risk

        # Diabetes risk variants
        diabetes_variants = [
            "rs7903146",  # TCF7L2
            "rs5219",     # KCNJ11
            "rs1801282"   # PPARG
        ]

        risk_count = sum(1 for v in variants if v.get("variant_id", "").split(":")[0] in diabetes_variants)
        adjusted_risk = base_risk * (1 + risk_count * 0.3)

        return {
            "risk_score": min(adjusted_risk, 1.0),
            "risk_category": "high" if adjusted_risk > 0.25 else "moderate" if adjusted_risk > 0.15 else "low",
            "contributing_variants": risk_count,
            "prevention_focus": "lifestyle_modification"
        }

    def _calculate_alzheimer_risk(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate Alzheimer's disease risk"""
        base_risk = 0.12  # Base population risk

        # Alzheimer's risk variants
        alzheimer_variants = [
            "rs7412",     # APOE
            "rs429358",   # APOE
            "rs6656401"   # CR1
        ]

        risk_count = sum(1 for v in variants if v.get("variant_id", "").split(":")[0] in alzheimer_variants)
        adjusted_risk = base_risk * (1 + risk_count * 0.4)

        return {
            "risk_score": min(adjusted_risk, 1.0),
            "risk_category": "high" if adjusted_risk > 0.35 else "moderate" if adjusted_risk > 0.2 else "low",
            "contributing_variants": risk_count,
            "cognitive_assessment": "recommended" if adjusted_risk > 0.25 else "optional"
        }
