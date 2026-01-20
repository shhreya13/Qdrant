"""
Enhanced Crisis Memory & Recommendation Engine
Production-ready version with advanced features
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================
# ENUMS FOR TYPE SAFETY
# ============================================

class CrisisType(Enum):
    SUICIDE = "suicide"
    PANIC = "panic"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SUBSTANCE = "substance"
    SELF_HARM = "self_harm"
    TRAUMA = "trauma"
    EATING_DISORDER = "eating_disorder"
    PSYCHOSIS = "psychosis"

class AgeGroup(Enum):
    CHILD = "child"  # 0-12
    TEEN = "teen"  # 13-19
    YOUNG_ADULT = "young_adult"  # 20-29
    ADULT = "adult"  # 30-59
    ELDERLY = "elderly"  # 60+

class Outcome(Enum):
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    ESCALATED = "escalated"
    TRANSFERRED = "transferred"
    FOLLOWUP_NEEDED = "followup_needed"

class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ============================================
# DATA CLASSES FOR STRUCTURED DATA
# ============================================

@dataclass
class CaseMetadata:
    """Additional metadata for each case"""
    operator_id: str
    timestamp: str
    call_duration_minutes: int
    urgency_level: int
    location_type: str  # "urban", "rural", "suburban"
    previous_caller: bool
    language: str = "english"
    
@dataclass
class PerformanceMetrics:
    """Track performance of recommendations"""
    recommendation_id: str
    was_used: bool
    effectiveness_score: Optional[int] = None  # 1-5
    operator_feedback: Optional[str] = None
    timestamp: str = None


# ============================================
# ENHANCED VECTOR GENERATION
# ============================================

class VectorGenerator:
    """Generate more sophisticated vectors for cases"""
    
    def __init__(self, vector_size: int = 128):
        self.vector_size = vector_size
        
        # Enhanced base vectors with higher dimensionality
        self.crisis_bases = {
            CrisisType.SUICIDE: self._generate_base([0.9, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3]),
            CrisisType.PANIC: self._generate_base([0.1, 0.9, 0.1, 0.1, 0.4, 0.1, 0.2, 0.1]),
            CrisisType.DOMESTIC_VIOLENCE: self._generate_base([0.1, 0.1, 0.9, 0.1, 0.3, 0.5, 0.1, 0.2]),
            CrisisType.SUBSTANCE: self._generate_base([0.1, 0.1, 0.1, 0.9, 0.2, 0.1, 0.4, 0.1]),
            CrisisType.SELF_HARM: self._generate_base([0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4]),
            CrisisType.TRAUMA: self._generate_base([0.2, 0.3, 0.4, 0.1, 0.7, 0.2, 0.1, 0.5]),
            CrisisType.EATING_DISORDER: self._generate_base([0.3, 0.2, 0.1, 0.2, 0.1, 0.1, 0.8, 0.2]),
            CrisisType.PSYCHOSIS: self._generate_base([0.2, 0.4, 0.1, 0.3, 0.1, 0.2, 0.1, 0.9]),
        }
        
        self.age_modifiers = {
            AgeGroup.CHILD: [0.15, -0.05, 0.0, 0.0, 0.1, 0.0, 0.05, 0.0],
            AgeGroup.TEEN: [0.1, 0.05, 0.0, 0.05, 0.05, 0.0, 0.1, 0.0],
            AgeGroup.YOUNG_ADULT: [0.05, 0.1, 0.05, 0.1, 0.0, 0.05, 0.05, 0.05],
            AgeGroup.ADULT: [0.0, 0.05, 0.1, 0.05, 0.0, 0.1, 0.0, 0.0],
            AgeGroup.ELDERLY: [0.0, -0.05, 0.05, 0.0, 0.05, 0.05, 0.0, 0.1],
        }
    
    def _generate_base(self, seed: List[float]) -> List[float]:
        """Expand seed vector to full size"""
        base = []
        for i in range(self.vector_size):
            idx = i % len(seed)
            variation = (i // len(seed)) * 0.01
            base.append(seed[idx] + variation)
        return self._normalize(base)
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        magnitude = sum(v**2 for v in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [v / magnitude for v in vector]
    
    def generate(self, crisis_type: CrisisType, age_group: AgeGroup, 
                 triggers: List[str], urgency: int) -> List[float]:
        """Generate enhanced vector"""
        
        # Start with crisis base
        vector = self.crisis_bases[crisis_type].copy()
        
        # Apply age modification
        age_mod = self.age_modifiers[age_group]
        expanded_age_mod = self._generate_base(age_mod)
        vector = [v + a * 0.3 for v, a in zip(vector, expanded_age_mod)]
        
        # Apply trigger-based variations
        trigger_signature = self._trigger_signature(triggers)
        vector = [v + t * 0.2 for v, t in zip(vector, trigger_signature)]
        
        # Apply urgency weighting
        urgency_weight = urgency / 4.0  # Normalize to 0-1
        urgency_mod = [urgency_weight * 0.1] * self.vector_size
        vector = [v + u for v, u in zip(vector, urgency_mod)]
        
        return self._normalize(vector)
    
    def _trigger_signature(self, triggers: List[str]) -> List[float]:
        """Create a signature from triggers"""
        signature = [0.0] * self.vector_size
        for trigger in triggers:
            hash_val = hash(trigger.lower())
            for i in range(self.vector_size):
                signature[i] += ((hash_val >> i) & 1) * 0.01
        return self._normalize(signature)


# ============================================
# ENHANCED CASE MANAGEMENT
# ============================================

class CaseManager:
    """Manage crisis cases with enhanced features"""
    
    def __init__(self, vector_generator: VectorGenerator):
        self.vector_gen = vector_generator
    
    def create_case(
        self,
        crisis_type: str,
        age_group: str,
        triggers: List[str],
        protocols_used: List[str],
        outcome: str,
        what_worked: List[str],
        feedback: str,
        metadata: Dict,
        urgency_level: int = 2
    ) -> Dict:
        """Create a comprehensive case record"""
        
        # Convert strings to enums
        crisis_enum = CrisisType(crisis_type)
        age_enum = AgeGroup(age_group)
        outcome_enum = Outcome(outcome)
        
        # Generate vector
        vector = self.vector_gen.generate(
            crisis_enum, age_enum, triggers, urgency_level
        )
        
        case_id = str(uuid.uuid4())
        
        return {
            'case_id': case_id,
            'vector': vector,
            'crisis_type': crisis_type,
            'age_group': age_group,
            'triggers': triggers,
            'protocols_used': protocols_used,
            'outcome': outcome,
            'what_worked': what_worked,
            'what_didnt_work': metadata.get('what_didnt_work', []),
            'feedback': feedback,
            'urgency_level': urgency_level,
            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
            'operator_id': metadata.get('operator_id', 'unknown'),
            'call_duration_minutes': metadata.get('call_duration_minutes', 0),
            'location_type': metadata.get('location_type', 'unknown'),
            'previous_caller': metadata.get('previous_caller', False),
            'language': metadata.get('language', 'english'),
            'follow_up_needed': metadata.get('follow_up_needed', False),
            'resources_provided': metadata.get('resources_provided', []),
        }


# ============================================
# ENHANCED DATABASE CONNECTION
# ============================================

class QdrantManager:
    """Manage Qdrant connection and operations"""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "crisis_cases_v2"):
        self.client = QdrantClient(
            url="https://bf15866f-7a66-43fa-b798-c06cbb11105d.europe-west3-0.gcp.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9E6rJx6xEoBoM8sEb-UMizFpKGLK38x99XElVUg23g",
            )
        self.collection_name = collection_name
        self.vector_size = 128
    
    def setup_collection(self, recreate: bool = False):
        """Setup Qdrant collection with proper configuration"""
        
        if recreate:
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                print("✓ Cleared old data")
            except:
                pass
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection already exists or error: {e}")
    
    def save_case(self, case: Dict):
        """Save case with error handling"""
        try:
            point = PointStruct(
                id=case['case_id'],
                vector=case['vector'],
                payload={k: v for k, v in case.items() if k not in ['case_id', 'vector']}
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return True
        except Exception as e:
            print(f"Error saving case: {e}")
            return False
    
    def save_cases_batch(self, cases: List[Dict]):
        """Batch save for efficiency"""
        points = []
        for case in cases:
            point = PointStruct(
                id=case['case_id'],
                vector=case['vector'],
                payload={k: v for k, v in case.items() if k not in ['case_id', 'vector']}
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✓ Saved {len(cases)} cases")
            return True
        except Exception as e:
            print(f"Error in batch save: {e}")
            return False
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """Enhanced search with filtering"""
        
        search_params = {
            'collection_name': self.collection_name,
            'query': query_vector,
            'limit': limit,
            'score_threshold': score_threshold
        }
        
        # Add filters if provided
        if filters:
            # Build filter conditions
            conditions = []
            for field, value in filters.items():
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            if conditions:
                search_params['query_filter'] = Filter(must=conditions)
        
        try:
            response = self.client.query_points(**search_params)
            results = response.points
        except:
            # Fallback to old API
            results = self.client.search(**search_params)
        
        similar_cases = []
        for result in results:
            case = result.payload
            case['similarity_score'] = result.score
            similar_cases.append(case)
        
        return similar_cases
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_cases': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance
            }
        except:
            return {'error': 'Could not retrieve stats'}


# ============================================
# ADVANCED RECOMMENDATION ENGINE
# ============================================

class RecommendationEngine:
    """Generate intelligent recommendations from similar cases"""
    
    def __init__(self):
        self.weights = {
            'similarity': 0.4,
            'recency': 0.2,
            'success_rate': 0.25,
            'urgency_match': 0.15
        }
    
    def generate_recommendations(
        self,
        similar_cases: List[Dict],
        current_urgency: int,
        min_confidence: float = 0.6
    ) -> Dict:
        """Generate comprehensive recommendations"""
        
        if not similar_cases:
            return {
                'status': 'no_data',
                'message': 'No similar cases found',
                'confidence': 0.0
            }
        
        # Calculate weighted scores for each case
        scored_cases = self._score_cases(similar_cases, current_urgency)
        
        # Extract recommendations
        protocols = self._recommend_protocols(scored_cases)
        techniques = self._recommend_techniques(scored_cases)
        warnings = self._extract_warnings(scored_cases)
        resources = self._recommend_resources(scored_cases)
        
        # Calculate confidence
        confidence = self._calculate_confidence(scored_cases)
        
        # Success rate analysis
        success_analysis = self._analyze_success_rate(scored_cases)
        
        recommendation = {
            'status': 'success',
            'confidence': round(confidence, 2),
            'based_on_cases': len(similar_cases),
            'primary_protocol': protocols[0] if protocols else None,
            'alternative_protocols': protocols[1:3] if len(protocols) > 1 else [],
            'recommended_techniques': techniques[:3],
            'success_rate': success_analysis['overall_rate'],
            'success_by_outcome': success_analysis['by_outcome'],
            'warnings': warnings,
            'recommended_resources': resources,
            'case_summaries': self._summarize_cases(scored_cases[:3]),
            'follow_up_recommendation': self._recommend_follow_up(scored_cases)
        }
        
        return recommendation
    
    def _score_cases(self, cases: List[Dict], current_urgency: int) -> List[Dict]:
        """Score cases based on multiple factors"""
        scored = []
        
        for case in cases:
            # Similarity score (already provided)
            sim_score = case.get('similarity_score', 0.5)
            
            # Recency score
            try:
                timestamp = datetime.fromisoformat(case.get('timestamp', '2020-01-01'))
                days_old = (datetime.now() - timestamp).days
                recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
            except:
                recency_score = 0.5
            
            # Success score
            outcome = case.get('outcome', 'partial')
            success_score = {
                'successful': 1.0,
                'partial': 0.6,
                'transferred': 0.5,
                'escalated': 0.3,
                'followup_needed': 0.7
            }.get(outcome, 0.5)
            
            # Urgency match score
            case_urgency = case.get('urgency_level', 2)
            urgency_diff = abs(current_urgency - case_urgency)
            urgency_score = max(0, 1 - (urgency_diff / 4))
            
            # Calculate weighted total
            total_score = (
                sim_score * self.weights['similarity'] +
                recency_score * self.weights['recency'] +
                success_score * self.weights['success_rate'] +
                urgency_score * self.weights['urgency_match']
            )
            
            case['total_score'] = total_score
            scored.append(case)
        
        # Sort by total score
        scored.sort(key=lambda x: x['total_score'], reverse=True)
        return scored
    
    def _recommend_protocols(self, scored_cases: List[Dict]) -> List[str]:
        """Recommend protocols based on weighted cases"""
        protocol_scores = {}
        
        for case in scored_cases:
            weight = case['total_score']
            for protocol in case.get('protocols_used', []):
                protocol_scores[protocol] = protocol_scores.get(protocol, 0) + weight
        
        # Sort by score
        sorted_protocols = sorted(
            protocol_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [p[0] for p in sorted_protocols]
    
    def _recommend_techniques(self, scored_cases: List[Dict]) -> List[Dict]:
        """Recommend specific techniques with context"""
        technique_data = {}
        
        for case in scored_cases:
            weight = case['total_score']
            for technique in case.get('what_worked', []):
                if technique not in technique_data:
                    technique_data[technique] = {
                        'score': 0,
                        'count': 0,
                        'contexts': []
                    }
                technique_data[technique]['score'] += weight
                technique_data[technique]['count'] += 1
                technique_data[technique]['contexts'].append({
                    'crisis_type': case.get('crisis_type'),
                    'outcome': case.get('outcome')
                })
        
        # Convert to list and sort
        techniques = []
        for tech, data in technique_data.items():
            techniques.append({
                'technique': tech,
                'confidence': round(data['score'] / len(scored_cases), 2),
                'used_in_cases': data['count'],
                'success_contexts': data['contexts'][:2]
            })
        
        techniques.sort(key=lambda x: x['confidence'], reverse=True)
        return techniques
    
    def _extract_warnings(self, scored_cases: List[Dict]) -> List[str]:
        """Extract warnings from cases that escalated"""
        warnings = []
        
        for case in scored_cases:
            if case.get('outcome') == 'escalated':
                if case.get('what_didnt_work'):
                    warnings.extend([
                        f"⚠️ Avoid: {item} (led to escalation)"
                        for item in case['what_didnt_work']
                    ])
        
        return list(set(warnings))[:3]  # Unique, top 3
    
    def _recommend_resources(self, scored_cases: List[Dict]) -> List[str]:
        """Recommend resources based on successful cases"""
        resource_count = {}
        
        for case in scored_cases:
            if case.get('outcome') == 'successful':
                for resource in case.get('resources_provided', []):
                    resource_count[resource] = resource_count.get(resource, 0) + 1
        
        sorted_resources = sorted(
            resource_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [r[0] for r in sorted_resources[:5]]
    
    def _analyze_success_rate(self, scored_cases: List[Dict]) -> Dict:
        """Detailed success rate analysis"""
        outcomes = {}
        for case in scored_cases:
            outcome = case.get('outcome', 'unknown')
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        total = len(scored_cases)
        successful = outcomes.get('successful', 0)
        
        return {
            'overall_rate': round((successful / total * 100) if total > 0 else 0, 1),
            'by_outcome': {
                outcome: round((count / total * 100), 1)
                for outcome, count in outcomes.items()
            }
        }
    
    def _calculate_confidence(self, scored_cases: List[Dict]) -> float:
        """Calculate overall confidence in recommendations"""
        if not scored_cases:
            return 0.0
        
        # Factors affecting confidence
        num_cases = len(scored_cases)
        avg_similarity = sum(c.get('similarity_score', 0) for c in scored_cases) / num_cases
        avg_total_score = sum(c.get('total_score', 0) for c in scored_cases) / num_cases
        
        # More cases = higher confidence (diminishing returns)
        case_confidence = min(1.0, num_cases / 10)
        
        # Combine factors
        confidence = (
            avg_similarity * 0.4 +
            avg_total_score * 0.4 +
            case_confidence * 0.2
        )
        
        return min(1.0, confidence)
    
    def _summarize_cases(self, cases: List[Dict]) -> List[Dict]:
        """Create brief summaries of top cases"""
        summaries = []
        for case in cases:
            summaries.append({
                'crisis_type': case.get('crisis_type'),
                'age_group': case.get('age_group'),
                'outcome': case.get('outcome'),
                'similarity': round(case.get('similarity_score', 0), 3),
                'key_technique': case.get('what_worked', ['N/A'])[0] if case.get('what_worked') else 'N/A',
                'note': case.get('feedback', '')[:100]
            })
        return summaries
    
    def _recommend_follow_up(self, scored_cases: List[Dict]) -> str:
        """Recommend whether follow-up is needed"""
        follow_up_count = sum(
            1 for c in scored_cases
            if c.get('follow_up_needed', False)
        )
        
        if follow_up_count > len(scored_cases) / 2:
            return "RECOMMENDED - Similar cases often required follow-up"
        elif follow_up_count > 0:
            return "CONSIDER - Some similar cases needed follow-up"
        else:
            return "OPTIONAL - Similar cases typically resolved in single session"


# ============================================
# ANALYTICS AND REPORTING
# ============================================

class AnalyticsEngine:
    """Analyze patterns in crisis data"""
    
    def __init__(self, db_manager: QdrantManager):
        self.db = db_manager
    
    def get_crisis_trends(self, days: int = 30) -> Dict:
        """Analyze crisis trends over time"""
        # This would require timestamp-based queries
        # Simplified version for demonstration
        return {
            'total_cases': 'N/A - requires timestamp filtering',
            'most_common_crisis': 'N/A',
            'trend': 'N/A'
        }
    
    def get_protocol_effectiveness(self) -> Dict:
        """Analyze which protocols work best"""
        return {
            'analysis': 'Requires aggregation of all cases',
            'top_protocols': []
        }
    
    def get_operator_performance(self, operator_id: str) -> Dict:
        """Analyze individual operator performance"""
        return {
            'operator_id': operator_id,
            'total_cases': 'N/A',
            'success_rate': 'N/A'
        }


# ============================================
# MAIN ORCHESTRATOR
# ============================================

class CrisisAssistantSystem:
    """Main system orchestrating all components"""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        self.vector_gen = VectorGenerator(vector_size=128)
        self.case_manager = CaseManager(self.vector_gen)
        self.db_manager = QdrantManager(qdrant_url, qdrant_api_key)
        self.recommender = RecommendationEngine()
        self.analytics = AnalyticsEngine(self.db_manager)
    
    def initialize(self, recreate_db: bool = False):
        """Initialize the system"""
        print("🚀 Initializing Crisis Assistant System...")
        self.db_manager.setup_collection(recreate=recreate_db)
        stats = self.db_manager.get_collection_stats()
        print(f"📊 Database stats: {stats}")
    
    def add_case(self, **kwargs) -> bool:
        """Add a new case to the system"""
        case = self.case_manager.create_case(**kwargs)
        return self.db_manager.save_case(case)
    
    def add_cases_batch(self, cases_data: List[Dict]) -> bool:
        """Add multiple cases at once"""
        cases = [self.case_manager.create_case(**c) for c in cases_data]
        return self.db_manager.save_cases_batch(cases)
    
    def get_recommendations(
        self,
        crisis_type: str,
        age_group: str,
        triggers: List[str],
        urgency_level: int = 2,
        filters: Optional[Dict] = None,
        num_similar: int = 5
    ) -> Dict:
        """Get recommendations for current crisis"""
        
        # Generate query vector
        crisis_enum = CrisisType(crisis_type)
        age_enum = AgeGroup(age_group)
        query_vector = self.vector_gen.generate(
            crisis_enum, age_enum, triggers, urgency_level
        )
        
        # Search for similar cases
        similar_cases = self.db_manager.search_similar(
            query_vector,
            limit=num_similar,
            filters=filters
        )
        
        # Generate recommendations
        recommendations = self.recommender.generate_recommendations(
            similar_cases,
            urgency_level
        )
        
        return recommendations
    
    def print_recommendations(self, recommendations: Dict):
        """Pretty print recommendations"""
        print("\n" + "=" * 70)
        print("🤖 AI RECOMMENDATIONS")
        print("=" * 70)
        
        if recommendations['status'] != 'success':
            print(f"\n⚠️  {recommendations.get('message', 'No recommendations available')}")
            return
        
        print(f"\n📊 Confidence Level: {recommendations['confidence']*100:.1f}%")
        print(f"📚 Based on: {recommendations['based_on_cases']} similar cases")
        print(f"✅ Success Rate: {recommendations['success_rate']}%")
        
        print(f"\n🎯 PRIMARY PROTOCOL:")
        print(f"   → {recommendations['primary_protocol']}")
        
        if recommendations['alternative_protocols']:
            print(f"\n🔄 ALTERNATIVES:")
            for alt in recommendations['alternative_protocols']:
                print(f"   → {alt}")
        
        print(f"\n💡 RECOMMENDED TECHNIQUES:")
        for tech in recommendations['recommended_techniques']:
            print(f"   → {tech['technique']}")
            print(f"     Confidence: {tech['confidence']*100:.0f}% | Used in {tech['used_in_cases']} cases")
        
        if recommendations['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in recommendations['warnings']:
                print(f"   {warning}")
        
        if recommendations['recommended_resources']:
            print(f"\n📋 RESOURCES TO CONSIDER:")
            for resource in recommendations['recommended_resources']:
                print(f"   → {resource}")
        
        print(f"\n🔄 FOLLOW-UP: {recommendations['follow_up_recommendation']}")
        
        print("\n" + "=" * 70)


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Initialize system
    system = CrisisAssistantSystem(
        qdrant_url="https://bf15866f-7a66-43fa-b798-c06cbb11105d.europe-west3-0.gcp.cloud.qdrant.io:6333",
        qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9E6rJx6xEoBoM8sEb-UMizFpKGLK38x99XElVUg23g"
    )
    
    system.initialize(recreate_db=True)
    
    # Create enhanced sample cases
    print("\n📝 Adding sample cases...")
    
    sample_cases = [
        {
            'crisis_type': 'suicide',
            'age_group': 'teen',
            'triggers': ['pills', 'family conflict', 'school pressure'],
            'protocols_used': ['columbia-protocol', 'safety-planning'],
            'outcome': 'successful',
            'what_worked': ['direct questions', 'empathy', 'safety contract'],
            'feedback': 'Teen responded well to direct approach and safety planning',
            'urgency_level': 4,
            'metadata': {
                'operator_id': 'OP-001',
                'timestamp': '2025-01-15T14:30:00',
                'call_duration_minutes': 45,
                'location_type': 'suburban',
                'previous_caller': False,
                'what_didnt_work': [],
                'resources_provided': ['teen mental health hotline', 'family therapy referral'],
                'follow_up_needed': True
            }
        },
        {
            'crisis_type': 'suicide',
            'age_group': 'teen',
            'triggers': ['overdose', 'breakup', 'social media'],
            'protocols_used': ['columbia-protocol', 'emergency'],
            'outcome': 'escalated',
            'what_worked': ['quick assessment', 'emergency services'],
            'feedback': 'Called 911 immediately - pills already taken',
            'urgency_level': 4,
            'metadata': {
                'operator_id': 'OP-002',
                'timestamp': '2025-01-14T22:15:00',
                'call_duration_minutes': 12,
                'location_type': 'urban',
                'previous_caller': False,
                'what_didnt_work': ['calming techniques - too late'],
                'resources_provided': ['emergency services'],
                'follow_up_needed': True
            }
        },
        {
            'crisis_type': 'panic',
            'age_group': 'adult',
            'triggers': ['chest pain', 'hyperventilating', 'work stress'],
            'protocols_used': ['grounding-technique', 'breathing-exercises'],
            'outcome': 'successful',
            'what_worked': ['breathing exercises', '5-4-3-2-1', 'reassurance'],
            'feedback': 'Caller calmed within 15 minutes using grounding',
            'urgency_level': 3,
            'metadata': {
                'operator_id': 'OP-001',
                'timestamp': '2025-01-16T09:45:00',
                'call_duration_minutes': 20,
                'location_type': 'urban',
                'previous_caller': True,
                'resources_provided': ['anxiety management app', 'therapy referral'],
                'follow_up_needed': False
            }
        },
        # Add more diverse cases
        {
            'crisis_type': 'self_harm',
            'age_group': 'teen',
            'triggers': ['cutting', 'emotional numbness', 'bullying'],
            'protocols_used': ['harm-reduction', 'safety-planning'],
            'outcome': 'successful',
            'what_worked': ['non-judgmental approach', 'coping alternatives', 'validation'],
            'feedback': 'Teen agreed to try alternative coping strategies',
            'urgency_level': 3,
            'metadata': {
                'operator_id': 'OP-003',
                'timestamp': '2025-01-15T16:00:00',
                'call_duration_minutes': 35,
                'location_type': 'suburban',
                'previous_caller': False,
                'resources_provided': ['DBT skills app', 'school counselor'],
                'follow_up_needed': True
            }
        }
    ]
    
    system.add_cases_batch(sample_cases)
    
    # Simulate current crisis
    print("\n" + "=" * 70)
    print("🚨 CURRENT CRISIS SCENARIO")
    print("=" * 70)
    
    current_crisis = {
        'crisis_type': 'suicide',
        'age_group': 'teen',
        'triggers': ['depression', 'isolation', 'academic failure'],
        'urgency_level': 3
    }
    
    print(f"\nCrisis Type: {current_crisis['crisis_type']}")
    print(f"Age Group: {current_crisis['age_group']}")
    print(f"Triggers: {', '.join(current_crisis['triggers'])}")
    print(f"Urgency: {current_crisis['urgency_level']}/4")
    
    # Get recommendations
    recommendations = system.get_recommendations(**current_crisis)
    
    # Display recommendations
    system.print_recommendations(recommendations)
    
    print("\n✅ System demo complete!")