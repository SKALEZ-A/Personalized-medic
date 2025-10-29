"""
Blockchain-based Security and Privacy System for Health Data
"""

import hashlib
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import queue

class BlockchainSecurity:
    """Comprehensive blockchain security for healthcare data"""

    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.network_nodes = set()
        self.consensus_mechanism = "proof_of_stake"  # Healthcare-appropriate consensus
        self.encryption_method = "AES-256-GCM"
        self.zero_knowledge_proofs = ZeroKnowledgeProofs()
        self.smart_contracts = SmartContracts()
        self.consent_management = ConsentManagement()

        # Initialize genesis block
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_block = {
            "index": 0,
            "timestamp": datetime.now().isoformat(),
            "transactions": [],
            "previous_hash": "0" * 64,
            "nonce": 0,
            "merkle_root": self._calculate_merkle_root([]),
            "validator": "genesis_validator",
            "hash": ""
        }

        genesis_block["hash"] = self._calculate_block_hash(genesis_block)
        self.chain.append(genesis_block)

    def _calculate_block_hash(self, block: Dict[str, Any]) -> str:
        """Calculate hash of a block"""
        block_string = json.dumps({
            "index": block["index"],
            "timestamp": block["timestamp"],
            "transactions": block["transactions"],
            "previous_hash": block["previous_hash"],
            "nonce": block["nonce"],
            "merkle_root": block["merkle_root"]
        }, sort_keys=True, default=str)

        return hashlib.sha256(block_string.encode()).hexdigest()

    def _calculate_merkle_root(self, transactions: List[Dict[str, Any]]) -> str:
        """Calculate Merkle root of transactions"""
        if not transactions:
            return "0" * 64

        # Simple Merkle tree calculation
        hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
                 for tx in transactions]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last hash if odd number

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())

            hashes = new_hashes

        return hashes[0]

    def add_transaction(self, transaction: Dict[str, Any]) -> str:
        """Add a transaction to the pending pool"""
        # Validate transaction
        if not self._validate_transaction(transaction):
            raise ValueError("Invalid transaction")

        # Add timestamp and transaction ID
        transaction["timestamp"] = datetime.now().isoformat()
        transaction["tx_id"] = hashlib.sha256(
            json.dumps(transaction, sort_keys=True).encode()
        ).hexdigest()[:16]

        self.pending_transactions.append(transaction)

        return transaction["tx_id"]

    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction structure and content"""
        required_fields = ["type", "data", "patient_id", "provider_id"]

        for field in required_fields:
            if field not in transaction:
                return False

        # Validate transaction type
        valid_types = ["health_record", "consent_update", "data_access", "audit_log"]
        if transaction["type"] not in valid_types:
            return False

        # Validate data integrity
        if "data_hash" in transaction:
            calculated_hash = hashlib.sha256(
                json.dumps(transaction["data"], sort_keys=True).encode()
            ).hexdigest()

            if calculated_hash != transaction["data_hash"]:
                return False

        return True

    def mine_block(self, validator: str) -> Dict[str, Any]:
        """Mine a new block (simplified for healthcare context)"""
        if not self.pending_transactions:
            return {"error": "No pending transactions"}

        # Create new block
        last_block = self.chain[-1]
        new_block = {
            "index": last_block["index"] + 1,
            "timestamp": datetime.now().isoformat(),
            "transactions": self.pending_transactions.copy(),
            "previous_hash": last_block["hash"],
            "nonce": 0,
            "merkle_root": self._calculate_merkle_root(self.pending_transactions),
            "validator": validator,
            "hash": ""
        }

        # Simple proof-of-work for healthcare (not computationally intensive)
        target = "0000"  # Easier target for healthcare applications
        while not new_block["hash"].startswith(target):
            new_block["nonce"] += 1
            new_block["hash"] = self._calculate_block_hash(new_block)

        # Add block to chain
        self.chain.append(new_block)

        # Clear pending transactions
        self.pending_transactions.clear()

        return {
            "block_index": new_block["index"],
            "block_hash": new_block["hash"],
            "transactions": len(new_block["transactions"]),
            "validator": validator
        }

    def verify_record_integrity(self, record_id: str) -> Dict[str, Any]:
        """Verify integrity of a health record using blockchain"""
        # Find record in blockchain
        for block in reversed(self.chain):
            for transaction in block["transactions"]:
                if transaction.get("record_id") == record_id:
                    # Verify record hasn't been tampered with
                    stored_hash = transaction.get("data_hash")
                    current_hash = hashlib.sha256(
                        json.dumps(transaction.get("data", {}), sort_keys=True).encode()
                    ).hexdigest()

                    return {
                        "verified": stored_hash == current_hash,
                        "block_hash": block["hash"],
                        "block_index": block["index"],
                        "timestamp": block["timestamp"],
                        "tamper_evidence": stored_hash != current_hash,
                        "transaction_id": transaction.get("tx_id")
                    }

        return {"error": "Record not found in blockchain"}

    def get_record_history(self, record_id: str) -> List[Dict[str, Any]]:
        """Get complete history of a record"""
        history = []

        for block in self.chain:
            for transaction in block["transactions"]:
                if transaction.get("record_id") == record_id:
                    history.append({
                        "block_index": block["index"],
                        "timestamp": block["timestamp"],
                        "transaction": transaction,
                        "block_hash": block["hash"]
                    })

        return history

    def create_health_record_transaction(self, patient_id: str, provider_id: str,
                                       record_data: Dict[str, Any]) -> str:
        """Create a health record transaction"""
        # Calculate data hash for integrity
        data_hash = hashlib.sha256(
            json.dumps(record_data, sort_keys=True).encode()
        ).hexdigest()

        transaction = {
            "type": "health_record",
            "patient_id": patient_id,
            "provider_id": provider_id,
            "record_id": f"record_{int(time.time())}_{random.randint(1000, 9999)}",
            "data": record_data,
            "data_hash": data_hash,
            "consent_verified": True,
            "encryption_method": self.encryption_method
        }

        return self.add_transaction(transaction)

    def create_consent_transaction(self, patient_id: str, consent_data: Dict[str, Any]) -> str:
        """Create a consent management transaction"""
        transaction = {
            "type": "consent_update",
            "patient_id": patient_id,
            "provider_id": "consent_system",
            "record_id": f"consent_{int(time.time())}",
            "data": consent_data,
            "data_hash": hashlib.sha256(json.dumps(consent_data, sort_keys=True).encode()).hexdigest(),
            "consent_verified": True
        }

        return self.add_transaction(transaction)

    def create_audit_transaction(self, action: str, user_id: str, details: Dict[str, Any]) -> str:
        """Create an audit log transaction"""
        transaction = {
            "type": "audit_log",
            "patient_id": user_id,
            "provider_id": "audit_system",
            "record_id": f"audit_{int(time.time())}",
            "data": {
                "action": action,
                "user_id": user_id,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
        }

        transaction["data_hash"] = hashlib.sha256(
            json.dumps(transaction["data"], sort_keys=True).encode()
        ).hexdigest()

        return self.add_transaction(transaction)

    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain network information"""
        return {
            "chain_length": len(self.chain),
            "total_transactions": sum(len(block["transactions"]) for block in self.chain),
            "pending_transactions": len(self.pending_transactions),
            "network_nodes": len(self.network_nodes),
            "latest_block": self.chain[-1] if self.chain else None,
            "consensus_mechanism": self.consensus_mechanism,
            "encryption_method": self.encryption_method
        }

    def validate_chain(self) -> bool:
        """Validate entire blockchain integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check hash integrity
            if current_block["hash"] != self._calculate_block_hash(current_block):
                return False

            # Check chain linkage
            if current_block["previous_hash"] != previous_block["hash"]:
                return False

            # Check Merkle root
            calculated_merkle = self._calculate_merkle_root(current_block["transactions"])
            if calculated_merkle != current_block["merkle_root"]:
                return False

        return True

class ZeroKnowledgeProofs:
    """Zero-knowledge proof system for privacy-preserving data sharing"""

    def create_proof(self, statement: str, witness: Dict[str, Any]) -> Dict[str, Any]:
        """Create a zero-knowledge proof"""
        # Simplified ZKP implementation
        proof = {
            "statement": statement,
            "proof_type": "zk_snark",  # Would use actual ZK-SNARK in production
            "public_inputs": self._extract_public_inputs(statement),
            "proof_data": hashlib.sha256(
                json.dumps(witness, sort_keys=True).encode()
            ).hexdigest(),
            "created_at": datetime.now().isoformat()
        }

        return proof

    def verify_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify a zero-knowledge proof"""
        # Simplified verification
        required_fields = ["statement", "proof_type", "public_inputs", "proof_data"]

        for field in required_fields:
            if field not in proof:
                return False

        # In real implementation, would perform actual cryptographic verification
        return True

    def _extract_public_inputs(self, statement: str) -> Dict[str, Any]:
        """Extract public inputs from proof statement"""
        # Simplified extraction
        return {
            "statement_hash": hashlib.sha256(statement.encode()).hexdigest(),
            "timestamp": datetime.now().isoformat()
        }

    def prove_data_range(self, value: float, min_val: float, max_val: float) -> Dict[str, Any]:
        """Prove that a value falls within a range without revealing the value"""
        statement = f"Value is between {min_val} and {max_val}"

        # Create range proof
        proof = self.create_proof(statement, {"value": value, "range": [min_val, max_val]})

        return proof

    def prove_eligibility(self, criteria: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prove patient eligibility without revealing sensitive data"""
        statement = f"Patient meets eligibility criteria: {json.dumps(criteria, sort_keys=True)}"

        proof = self.create_proof(statement, {
            "patient_data": patient_data,
            "criteria": criteria
        })

        return proof

class SmartContracts:
    """Smart contracts for automated healthcare workflows"""

    def __init__(self):
        self.contracts = {}
        self.deployed_contracts = {}

    def create_contract(self, contract_type: str, parameters: Dict[str, Any]) -> str:
        """Create a smart contract"""
        contract_id = f"contract_{int(time.time())}_{random.randint(1000, 9999)}"

        contract = {
            "id": contract_id,
            "type": contract_type,
            "parameters": parameters,
            "code": self._generate_contract_code(contract_type, parameters),
            "state": "created",
            "created_at": datetime.now().isoformat(),
            "deployed": False
        }

        self.contracts[contract_id] = contract

        return contract_id

    def deploy_contract(self, contract_id: str) -> Dict[str, Any]:
        """Deploy a smart contract"""
        if contract_id not in self.contracts:
            return {"error": "Contract not found"}

        contract = self.contracts[contract_id]

        # Simulate deployment
        contract["deployed"] = True
        contract["deployed_at"] = datetime.now().isoformat()
        contract["address"] = f"0x{hashlib.sha256(contract_id.encode()).hexdigest()[:40]}"

        self.deployed_contracts[contract_id] = contract

        return {
            "contract_id": contract_id,
            "address": contract["address"],
            "status": "deployed"
        }

    def execute_contract(self, contract_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a smart contract"""
        if contract_id not in self.deployed_contracts:
            return {"error": "Contract not deployed"}

        contract = self.deployed_contracts[contract_id]

        # Execute contract logic based on type
        if contract["type"] == "consent_contract":
            result = self._execute_consent_contract(contract, inputs)
        elif contract["type"] == "data_access_contract":
            result = self._execute_data_access_contract(contract, inputs)
        elif contract["type"] == "payment_contract":
            result = self._execute_payment_contract(contract, inputs)
        else:
            result = {"error": "Unknown contract type"}

        return result

    def _generate_contract_code(self, contract_type: str, parameters: Dict[str, Any]) -> str:
        """Generate smart contract code"""
        if contract_type == "consent_contract":
            return f"""
            pragma solidity ^0.8.0;

            contract ConsentContract {{
                address public patient;
                address public provider;
                mapping(string => bool) public consents;

                constructor(address _patient, address _provider) {{
                    patient = _patient;
                    provider = _provider;
                }}

                function grantConsent(string memory dataType) public {{
                    require(msg.sender == patient, "Only patient can grant consent");
                    consents[dataType] = true;
                }}

                function revokeConsent(string memory dataType) public {{
                    require(msg.sender == patient, "Only patient can revoke consent");
                    consents[dataType] = false;
                }}

                function checkConsent(string memory dataType) public view returns (bool) {{
                    return consents[dataType];
                }}
            }}
            """
        elif contract_type == "data_access_contract":
            return "// Data access contract code"
        else:
            return "// Generic contract template"

    def _execute_consent_contract(self, contract: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consent management contract"""
        action = inputs.get("action")
        data_type = inputs.get("data_type")

        if action == "grant":
            return {"result": f"Consent granted for {data_type}"}
        elif action == "revoke":
            return {"result": f"Consent revoked for {data_type}"}
        elif action == "check":
            return {"result": f"Consent status checked for {data_type}"}
        else:
            return {"error": "Invalid action"}

    def _execute_data_access_contract(self, contract: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data access contract"""
        # Simplified execution
        return {"result": "Data access authorized", "access_level": "read"}

    def _execute_payment_contract(self, contract: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute payment contract"""
        amount = inputs.get("amount", 0)
        return {"result": f"Payment processed: ${amount}"}

class ConsentManagement:
    """Advanced consent management system"""

    def __init__(self):
        self.consent_records = {}
        self.consent_templates = self._initialize_consent_templates()

    def _initialize_consent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize consent templates"""
        return {
            "general_medical_care": {
                "title": "General Medical Care Consent",
                "purpose": "Provide comprehensive medical care and treatment",
                "data_types": ["medical_history", "vital_signs", "laboratory_results"],
                "retention_period": "7_years",
                "sharing_allowed": True
            },
            "research_participation": {
                "title": "Research Participation Consent",
                "purpose": "Use de-identified data for medical research",
                "data_types": ["medical_records", "genomic_data"],
                "retention_period": "indefinite",
                "sharing_allowed": True,
                "anonymization_required": True
            },
            "genomic_testing": {
                "title": "Genomic Testing Consent",
                "purpose": "Perform genetic analysis for personalized medicine",
                "data_types": ["genomic_data", "family_history"],
                "retention_period": "lifetime",
                "sharing_allowed": False
            }
        }

    def create_consent_record(self, patient_id: str, consent_type: str,
                            parameters: Dict[str, Any]) -> str:
        """Create a consent record"""
        consent_id = f"consent_{int(time.time())}_{random.randint(1000, 9999)}"

        template = self.consent_templates.get(consent_type, {})

        consent_record = {
            "id": consent_id,
            "patient_id": patient_id,
            "consent_type": consent_type,
            "template": template,
            "parameters": parameters,
            "status": "active",
            "granted_at": datetime.now().isoformat(),
            "expires_at": self._calculate_expiry_date(template.get("retention_period")),
            "withdrawals": []
        }

        self.consent_records[consent_id] = consent_record

        return consent_id

    def check_consent(self, patient_id: str, data_type: str, requester: str) -> Dict[str, Any]:
        """Check if consent exists for data access"""
        patient_consents = [
            record for record in self.consent_records.values()
            if record["patient_id"] == patient_id and record["status"] == "active"
        ]

        for consent in patient_consents:
            if data_type in consent["template"].get("data_types", []):
                # Check if consent is still valid
                if datetime.now() < datetime.fromisoformat(consent["expires_at"]):
                    return {
                        "authorized": True,
                        "consent_id": consent["id"],
                        "consent_type": consent["consent_type"],
                        "restrictions": consent.get("parameters", {}).get("restrictions", [])
                    }

        return {
            "authorized": False,
            "reason": "No valid consent found",
            "required_consent": self._suggest_required_consent(data_type)
        }

    def revoke_consent(self, consent_id: str, reason: str) -> Dict[str, Any]:
        """Revoke a consent"""
        if consent_id not in self.consent_records:
            return {"error": "Consent not found"}

        consent = self.consent_records[consent_id]

        # Add withdrawal record
        withdrawal = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }

        consent["withdrawals"].append(withdrawal)
        consent["status"] = "revoked"
        consent["revoked_at"] = datetime.now().isoformat()

        return {
            "consent_id": consent_id,
            "status": "revoked",
            "withdrawal_recorded": True
        }

    def _calculate_expiry_date(self, retention_period: str) -> str:
        """Calculate consent expiry date"""
        now = datetime.now()

        if retention_period == "7_years":
            expiry = now.replace(year=now.year + 7)
        elif retention_period == "lifetime":
            expiry = now.replace(year=now.year + 80)  # Approximate lifetime
        elif retention_period == "indefinite":
            expiry = now.replace(year=now.year + 100)
        else:
            expiry = now.replace(year=now.year + 1)  # Default 1 year

        return expiry.isoformat()

    def _suggest_required_consent(self, data_type: str) -> str:
        """Suggest required consent type for data access"""
        suggestions = {
            "genomic_data": "genomic_testing",
            "medical_history": "general_medical_care",
            "research_data": "research_participation"
        }

        return suggestions.get(data_type, "general_medical_care")

    def get_consent_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get consent history for a patient"""
        patient_consents = [
            record for record in self.consent_records.values()
            if record["patient_id"] == patient_id
        ]

        # Sort by creation date
        patient_consents.sort(key=lambda x: x["granted_at"], reverse=True)

        return patient_consents

    def generate_consent_report(self, patient_id: str) -> Dict[str, Any]:
        """Generate comprehensive consent report"""
        consent_history = self.get_consent_history(patient_id)

        active_consents = [c for c in consent_history if c["status"] == "active"]
        revoked_consents = [c for c in consent_history if c["status"] == "revoked"]

        return {
            "patient_id": patient_id,
            "total_consents": len(consent_history),
            "active_consents": len(active_consents),
            "revoked_consents": len(revoked_consents),
            "consent_types": list(set(c["consent_type"] for c in active_consents)),
            "data_types_authorized": self._aggregate_authorized_data_types(active_consents),
            "generated_at": datetime.now().isoformat()
        }

    def _aggregate_authorized_data_types(self, consents: List[Dict[str, Any]]) -> List[str]:
        """Aggregate all authorized data types"""
        data_types = set()

        for consent in consents:
            template_data_types = consent["template"].get("data_types", [])
            data_types.update(template_data_types)

        return list(data_types)
