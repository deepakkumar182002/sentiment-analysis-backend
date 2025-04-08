from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
analyzer = SentimentIntensityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        logger.info("Received sentiment analysis request")
        
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        feedbacks = data.get("feedbacks", [])
        
        if not feedbacks:
            logger.error("No feedbacks provided")
            return jsonify({"error": "No feedbacks provided"}), 400
        
        logger.info(f"Analyzing {len(feedbacks)} feedbacks")
        
        sentiments = []
        sentiment_details = []
        
        # Calculate overall statistics
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        total_compound = 0
        
        for i, feedback in enumerate(feedbacks):
            try:
                logger.info(f"Analyzing feedback {i+1}/{len(feedbacks)}: {feedback[:50]}...")
                
                # Get VADER sentiment scores
                scores = analyzer.polarity_scores(feedback)
                compound = scores['compound']
                
                # Determine sentiment based on compound score
                if compound >= 0.05:
                    sentiment = "Positive"
                    total_positive += 1
                elif compound <= -0.05:
                    sentiment = "Negative"
                    total_negative += 1
                else:
                    sentiment = "Neutral"
                    total_neutral += 1
                
                total_compound += compound
                
                logger.info(f"Feedback {i+1} sentiment: {sentiment} (compound: {compound})")
                sentiments.append(sentiment)
                
                # Add detailed sentiment information
                sentiment_details.append({
                    "text": feedback,
                    "sentiment": sentiment,
                    "compound_score": compound,
                    "confidence": round(abs(compound) * 100, 2),
                    "scores": {
                        "pos": scores['pos'],
                        "neg": scores['neg'],
                        "neu": scores['neu'],
                        "compound": scores['compound']
                    }
                })
                
            except Exception as e:
                logger.error(f"Error analyzing feedback {i+1}: {str(e)}")
                sentiments.append("Error")
                sentiment_details.append({
                    "text": feedback,
                    "sentiment": "Error",
                    "compound_score": 0,
                    "confidence": 0,
                    "scores": {
                        "pos": 0,
                        "neg": 0,
                        "neu": 0,
                        "compound": 0
                    }
                })
        
        # Calculate overall percentages
        total = len(feedbacks)
        positive_percentage = (total_positive / total) * 100 if total > 0 else 0
        negative_percentage = (total_negative / total) * 100 if total > 0 else 0
        neutral_percentage = (total_neutral / total) * 100 if total > 0 else 0
        
        # Calculate average compound score
        avg_compound_score = total_compound / total if total > 0 else 0
        
        # Determine overall sentiment
        if avg_compound_score > 0.05:
            overall_sentiment = "Positive"
        elif avg_compound_score < -0.05:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        logger.info(f"Analysis complete. Results: {sentiments}")
        
        return jsonify({
            "sentiments": sentiments,
            "details": sentiment_details,
            "summary": {
                "positive_count": total_positive,
                "negative_count": total_negative,
                "neutral_count": total_neutral,
                "positive_percentage": positive_percentage,
                "negative_percentage": negative_percentage,
                "neutral_percentage": neutral_percentage,
                "average_compound_score": avg_compound_score,
                "overall_sentiment": overall_sentiment
            }
        })
    
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    logger.info("Starting sentiment analysis server...")
    app.run(debug=True, host='127.0.0.1', port=5000) 