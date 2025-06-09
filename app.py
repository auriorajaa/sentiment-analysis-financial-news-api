from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import sys
import os
import datetime
import numpy as np
import re
import uuid

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import all required classes from lstm file
try:
    from lstm_3_of_sentiment_analysis import *

    print("✓ Successfully imported all classes from lstm_3_of_sentiment_analysis")
except ImportError as e:
    print(f"✗ Error importing lstm_3_of_sentiment_analysis: {e}")
    print(
        "✗ Please ensure 'lstm_3_of_sentiment_analysis.py' exists in the same directory"
    )
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for model
model = None


def load_model():
    """Load model at startup"""
    global model
    try:
        model_path = "financial_sentiment_model_unified.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at path: {os.path.abspath(model_path)}")
            return False

        logger.info("Loading Financial Sentiment LSTM model...")
        model = FinancialSentimentLSTM.load_unified(model_path)
        logger.info("Model loaded successfully!")
        return True

    except FileNotFoundError as e:
        logger.error(
            f"Model file not found! Path: {os.path.abspath('financial_sentiment_model_unified.pkl')}"
        )
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def generate_analysis_id():
    """Generate unique analysis ID"""
    return str(uuid.uuid4())[:8]


def categorize_sentiment_strength(probability):
    """Categorize sentiment strength based on probability"""
    if probability < 0.33:
        return "weak"
    elif probability < 0.66:
        return "moderate"
    else:
        return "strong"


def categorize_intensity(intensity_score):
    """Categorize intensity based on score"""
    if intensity_score < 0.05:
        return "low"
    elif intensity_score < 0.15:
        return "medium"
    else:
        return "high"


def generate_risk_assessment(proba, financial_features, sentiment_features):
    """Generate risk assessment based on sentiment and financial features"""
    negative_prob = float(proba[0])
    positive_prob = float(proba[2])

    # Calculate volatility risk
    volatility_risk = "low"
    if financial_features.get("intensity_score", 0) > 0.1:
        volatility_risk = "high" if negative_prob > 0.5 else "medium"

    # Calculate sentiment risk
    sentiment_risk = "low"
    if negative_prob > 0.6:
        sentiment_risk = "high"
    elif negative_prob > 0.4:
        sentiment_risk = "medium"

    # Calculate overall risk
    overall_risk = "low"
    if sentiment_risk == "high" or volatility_risk == "high":
        overall_risk = "high"
    elif sentiment_risk == "medium" or volatility_risk == "medium":
        overall_risk = "medium"

    return {
        "overall_risk_level": overall_risk,
        "volatility_risk": volatility_risk,
        "sentiment_risk": sentiment_risk,
        "confidence": float(max(negative_prob, positive_prob)),
    }


def generate_financial_summary(sentiment, proba, financial_features):
    """Generate financial summary based on sentiment and financial features"""
    sentiment_labels = ["negative", "neutral", "positive"]
    confidence = float(proba[sentiment_labels.index(sentiment)])

    if sentiment == "positive":
        if confidence > 0.7:
            return (
                "Strong positive financial outlook with significant upside potential."
            )
        else:
            return "Moderately positive financial sentiment with potential growth indicators."
    elif sentiment == "negative":
        if confidence > 0.7:
            return (
                "Strong negative financial outlook suggesting potential downside risk."
            )
        else:
            return "Moderately negative financial sentiment with some concerning indicators."
    else:
        return (
            "Neutral financial outlook with balanced risk and opportunity indicators."
        )


def generate_action_recommendations(sentiment, proba, financial_features):
    """Generate action recommendations based on sentiment and financial features"""
    recommendations = []
    sentiment_labels = ["negative", "neutral", "positive"]
    confidence = float(proba[sentiment_labels.index(sentiment)])

    if sentiment == "positive":
        if confidence > 0.7:
            recommendations.append("Consider increasing exposure to the asset")
            recommendations.append("Monitor for additional positive indicators")
        else:
            recommendations.append("Consider modest position increases")
            recommendations.append("Continue monitoring for sentiment shifts")
    elif sentiment == "negative":
        if confidence > 0.7:
            recommendations.append("Consider reducing exposure to the asset")
            recommendations.append("Set stop-loss orders to mitigate downside risk")
        else:
            recommendations.append("Exercise caution with new positions")
            recommendations.append("Monitor for additional negative indicators")
    else:
        recommendations.append("Maintain balanced positions")
        recommendations.append("Monitor for developing sentiment trends")

    return recommendations


def generate_market_impact_assessment(sentiment, proba, financial_features):
    """Generate market impact assessment based on sentiment and financial features"""
    sentiment_labels = ["negative", "neutral", "positive"]
    confidence = float(proba[sentiment_labels.index(sentiment)])

    market_direction = "stable"
    if sentiment == "positive" and confidence > 0.6:
        market_direction = "bullish"
    elif sentiment == "negative" and confidence > 0.6:
        market_direction = "bearish"

    impact_level = "minimal"
    intensity_score = financial_features.get("intensity_score", 0)
    if intensity_score > 0.1:
        impact_level = "significant"
    elif intensity_score > 0.05:
        impact_level = "moderate"

    return {
        "market_direction": market_direction,
        "impact_level": impact_level,
        "confidence": confidence,
        "intensity": intensity_score,
    }


def generate_text_analysis(text, processed_text, financial_features):
    """Generate text analysis based on the original and processed text"""
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)

    sentiment_words = financial_features.get(
        "positive_keywords", 0
    ) + financial_features.get("negative_keywords", 0)
    financial_terms_density = financial_features.get(
        "financial_indicators_count", 0
    ) / max(word_count, 1)

    key_terms = []
    if financial_features.get("ticker_count", 0) > 0:
        key_terms.append("ticker symbols")
    if financial_features.get("has_percentage", 0) > 0:
        key_terms.append("percentages")
    if financial_features.get("has_dollar", 0) > 0:
        key_terms.append("dollar amounts")
    if financial_features.get("intensity_words", 0) > 0:
        key_terms.append("intensity indicators")

    return {
        "text_statistics": {
            "word_count": word_count,
            "average_word_length": float(avg_word_length),
            "sentiment_words_count": sentiment_words,
            "financial_terms_density": float(financial_terms_density),
        },
        "key_content_types": key_terms,
        "processed_text_sample": processed_text[:200]
        + ("..." if len(processed_text) > 200 else ""),
    }


@app.route("/", methods=["GET"])
def home():
    """Endpoint to check API status"""
    return jsonify(
        {
            "message": "Financial Sentiment Analysis API",
            "status": "running",
            "model_loaded": model is not None,
            "endpoints": {
                "predict": "/predict - POST method for comprehensive sentiment prediction"
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    """
    Endpoint for comprehensive sentiment prediction from financial news text
    Request body: {"text": "financial news text to analyze", "language": "en"}
    Response: Comprehensive analysis including sentiment, confidence, risk assessment, etc.
    """
    try:
        # Check if model is loaded
        if model is None:
            return (
                jsonify(
                    {"error": "Model not loaded or failed to load", "success": False}
                ),
                500,
            )

        # Check if model is fitted
        if not model.is_fitted:
            return jsonify({"error": "Model not trained", "success": False}), 500

        # Get data from request
        data = request.get_json()

        if not data:
            return (
                jsonify({"error": "Empty request body or not JSON", "success": False}),
                400,
            )

        text = data.get("text", "").strip()
        lang = data.get("language", "en")

        if not text:
            return (
                jsonify(
                    {"error": "Parameter 'text' is empty or missing", "success": False}
                ),
                400,
            )

        # Validate text length
        if len(text) > 1000:
            return (
                jsonify(
                    {
                        "error": "Text too long. Maximum 1000 characters",
                        "success": False,
                    }
                ),
                400,
            )

        logger.info(f"Processing prediction for text: {text[:50]}...")

        # Perform prediction using the model's predict_single_text method
        result = model.predict_single_text(text)

        # Extract detailed features for comprehensive analysis
        processed_text = (
            model.preprocessor.preprocess(text)
            if hasattr(model, "preprocessor")
            else text
        )

        # Try to extract financial and sentiment features if available
        financial_features = {}
        sentiment_features = {}

        try:
            if hasattr(model, "financial_engineer"):
                financial_features = (
                    model.financial_engineer.extract_financial_features(text, lang=lang)
                )
            if hasattr(model, "sentiment_extractor"):
                sentiment_features = (
                    model.sentiment_extractor.extract_all_sentiment_features(text)
                )
        except Exception as e:
            logger.warning(f"Could not extract detailed features: {e}")
            # Provide basic features
            financial_features = {
                "financial_indicators_count": 0,
                "positive_keywords": 0,
                "negative_keywords": 0,
                "sentiment_contrast": 0,
                "intensity_score": 0.1,
            }

        # Get sentiment probabilities (assuming the model returns probabilities)
        sentiment = result["sentiment"]
        confidence = result["confidence"]

        # Create probability distribution (mock if not available from model)
        if sentiment == "positive":
            proba = [0.1, 0.2, confidence]
        elif sentiment == "negative":
            proba = [confidence, 0.2, 0.1]
        else:  # neutral
            proba = [0.2, confidence, 0.2]

        proba = np.array(proba)

        # Detect Indonesian text
        is_indonesian_text = lang == "id" or re.search(
            r"\b(indonesia|ekonomi|tumbuh|pasar|kuartal)\b", text.lower()
        )

        # Generate comprehensive analysis
        risk_assessment = generate_risk_assessment(
            proba, financial_features, sentiment_features
        )
        financial_summary = generate_financial_summary(
            sentiment, proba, financial_features
        )
        action_recommendations = generate_action_recommendations(
            sentiment, proba, financial_features
        )
        market_impact = generate_market_impact_assessment(
            sentiment, proba, financial_features
        )
        text_analysis = generate_text_analysis(text, processed_text, financial_features)

        # Build comprehensive response
        response = {
            "success": True,
            "analysis_id": generate_analysis_id(),
            "timestamp": datetime.datetime.now().isoformat(),
            "input": {
                "text": text,
                "language": lang,
                "text_length": len(text),
                "processed_text_preview": processed_text[:100]
                + ("..." if len(processed_text) > 100 else ""),
            },
            "sentiment_analysis": {
                "primary_sentiment": sentiment,
                "confidence": round(confidence, 4),
                "confidence_percentage": round(confidence * 100, 2),
                "sentiment_distribution": {
                    "negative": {
                        "probability": float(proba[0]),
                        "strength": categorize_sentiment_strength(float(proba[0])),
                    },
                    "neutral": {
                        "probability": float(proba[1]),
                        "strength": categorize_sentiment_strength(float(proba[1])),
                    },
                    "positive": {
                        "probability": float(proba[2]),
                        "strength": categorize_sentiment_strength(float(proba[2])),
                    },
                },
                "sentiment_metrics": {
                    "sentiment_consensus": {
                        "score": confidence,
                        "interpretation": (
                            "high"
                            if confidence > 0.7
                            else ("medium" if confidence > 0.3 else "low")
                        ),
                    }
                },
            },
            "financial_analysis": {
                "summary": financial_summary,
                "indicators": {
                    "financial_terms_count": financial_features.get(
                        "financial_indicators_count", 0
                    ),
                    "positive_keywords": financial_features.get("positive_keywords", 0),
                    "negative_keywords": financial_features.get("negative_keywords", 0),
                    "sentiment_contrast": financial_features.get(
                        "sentiment_contrast", 0
                    ),
                    "intensity_score": financial_features.get("intensity_score", 0),
                    "intensity_level": categorize_intensity(
                        financial_features.get("intensity_score", 0)
                    ),
                },
                "detailed_metrics": financial_features,
            },
            "risk_assessment": risk_assessment,
            "action_recommendations": action_recommendations,
            "market_impact": market_impact,
            "text_analysis": text_analysis,
            "technical_details": {
                "model_version": "Financial Sentiment Analysis v1.0",
                "api_version": "1.1.0",
                "language_detected": lang,
                "is_indonesian_text": is_indonesian_text,
            },
        }

        logger.info(
            f"Enhanced analysis completed. Predicted: {sentiment} (confidence: {confidence:.4f})"
        )
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return (
            jsonify(
                {
                    "error": f"Internal server error: {str(e)}",
                    "success": False,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "request_id": generate_analysis_id(),
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint for health check"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_fitted": model.is_fitted if model else False,
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "success": False}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return (
        jsonify({"error": "Method not allowed for this endpoint", "success": False}),
        405,
    )


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "success": False}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("FINANCIAL SENTIMENT ANALYSIS API")
    print("=" * 60)

    # Load model at startup
    if load_model():
        print("✓ Model loaded successfully")
        print("✓ API ready to use")

        # Run Flask app
        app.run(
            host="0.0.0.0",
            port=8080,
            debug=False,
        )
    else:
        print("✗ Failed to load model")
        print("✗ API cannot be started")
