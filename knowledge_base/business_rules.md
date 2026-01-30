# Business Rules Documentation

## Airbnb Analytics Business Logic

This document describes the business rules and domain-specific logic for the Airbnb Analytics platform.

---

## Review Sentiment Analysis

### BR-SEN-001: Sentiment Classification

**Business Context**: Review sentiment is used to calculate host performance scores, property rankings, and guest satisfaction metrics.

#### Classification Rules

| Sentiment | Definition | NLP Score Range |
|-----------|------------|-----------------|
| positive | Guest expresses satisfaction | > 0.6 |
| neutral | No strong opinion expressed | -0.2 to 0.6 |
| negative | Guest expresses dissatisfaction | < -0.2 |
| mixed | Contains both positive and negative | Dual-score detection |
| unknown | Cannot be classified | Confidence < 0.3 |

#### Business Impact

- **Host Rankings**: Only `positive`, `neutral`, `negative` count toward official rankings
- **Reporting**: `mixed` and `unknown` are included in exploratory analysis
- **Dashboard Display**: All sentiment types should be displayable

#### System Migration Note

The sentiment analysis system was upgraded in December 2024. During the transition:
- Legacy system produces: `positive`, `neutral`, `negative`
- New system also produces: `mixed`, `unknown`
- Both are valid and should be accepted in the data model

---

## Listing Room Types

### BR-RT-001: Room Type Classification

**Business Context**: Room types drive pricing recommendations, search filters, and capacity planning.

#### Standard Room Types

1. **Entire home/apt**
   - Guest has entire place to themselves
   - Includes houses, apartments, condos, lofts
   
2. **Private room**
   - Guest has private room but shares common areas
   - Must have lockable door
   
3. **Shared room**
   - Guest shares sleeping space with others
   - Budget option
   
4. **Hotel room**
   - Hotel or hostel accommodation
   - Professional hospitality setting

#### New Category: Studio

**Effective Date**: November 2024

**Definition**: A studio apartment is a self-contained unit where the living room, bedroom, and kitchen are combined into a single room.

**Classification Logic**:
- If listing is `Entire home/apt` AND
- Square footage < 500 sq ft AND
- Bedroom count = 0
- THEN classify as `Studio`

**Note for Analytics Engineering**: The `Studio` room type is now valid and should be added to accepted values. This helps differentiate compact urban listings for pricing analysis.

---

## Pricing Rules

### BR-PRC-001: Listing Price Guidelines

#### Standard Pricing

- **Floor Price**: $0.01 (minimum to prevent division errors)
- **Soft Cap**: $10,000/night (99th percentile)
- **Hard Cap**: $50,000/night (system maximum)

#### Exception Categories

| Category | Price Range | Validation |
|----------|-------------|------------|
| Standard | $0.01 - $10,000 | Always valid |
| Luxury | $10,000 - $50,000 | Requires luxury_flag = true |
| Promotional | $0.00 | Requires promo_campaign_id |
| Error | < $0.00 | Invalid, filter out |

#### Handling Out-of-Range Prices

For analytics purposes:
1. **$0.00 listings**: Include in count metrics, exclude from average calculations
2. **> $10,000 listings**: Include but flag for manual review
3. **Negative prices**: Data error - exclude entirely

---

## Host Data Rules

### BR-HOST-001: Host Account Lifecycle

#### Host ID Requirements

The `host_id` field is critical for attribution and must be present for:
- Active listings
- Completed bookings
- Published reviews

#### Exception: Migration Period

During host account migrations (consolidation, ownership transfer):
- Listings may temporarily have `NULL` host_id
- Grace period: 72 hours from migration start
- Records should NOT fail tests during this window

**Identification**: Check for records where:
- `host_id IS NULL`
- `_etl_loaded_at > CURRENT_TIMESTAMP - INTERVAL 72 HOURS`
- Status indicates migration

---

## Review Integrity Rules

### BR-REV-001: Review-Listing Relationship

#### Expected Behavior

Every review should have a corresponding active listing. However:

#### Known Exception Cases

1. **Deleted Listings**: A listing may be removed but historical reviews preserved
   - Reviews can exist for listing_id values not in current dim_listing
   - This is expected for compliance/historical record keeping
   
2. **Merged Listings**: When duplicate listings are merged:
   - Reviews retain original listing_id
   - Parent listing_id may not exist
   
3. **Test/Demo Data**: Development environment may have orphan reviews

#### Recommended Handling

```sql
-- For fact_reviews, use LEFT JOIN to preserve history
LEFT JOIN dim_listing ON fact_reviews.listing_id = dim_listing.listing_id

-- Add column to indicate listing status
CASE 
  WHEN dim_listing.listing_id IS NULL THEN 'listing_unavailable'
  ELSE 'active'
END AS listing_status
```

---

## Superhost Rules

### BR-HOST-002: Superhost Designation

**Definition**: A Superhost is an experienced host with excellent ratings.

#### Qualification Criteria
- Minimum 10 trips hosted in past year
- 90%+ response rate
- Less than 1% cancellation rate
- 4.8+ overall rating

#### Data Impact
- `is_superhost` column: 'true' or 'false' (string, not boolean)
- Updated quarterly
- Used in search ranking and badging

---

## Date and Time Rules

### BR-DATE-001: Review Date Constraints

**Valid Date Range**:
- **Minimum**: 2000-01-01 (Airbnb founding era)
- **Maximum**: Current date (no future reviews allowed)

**Future Date Handling**:
- Reviews with future dates are data errors
- Should be filtered or corrected at source
- If persistent, flag for ETL investigation

---

## Contact

For business rules clarification:
- **Product Owner**: product-analytics@company.com
- **Domain Expert**: domain-leads@company.com

