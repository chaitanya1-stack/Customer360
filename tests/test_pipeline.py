"""Small, hand-calculated fixtures validate accounting and time-boundary semantics."""
import sqlite3
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import SCHEMA,normalize
from src.features import extract_features,holdout,cohort_retention,wilson,add_personas,FEATURES
from src.modeling import FeatureTransform

ROOT=Path(__file__).resolve().parents[1]

class SQLTests(unittest.TestCase):
    def setUp(self):
        self.con=sqlite3.connect(":memory:");self.con.executescript(SCHEMA)
        rows=[
            ("1","12345","A",2,datetime(2011,1,1,12),10,100,"UK"),
            ("1","12346","B",1,datetime(2011,1,1,12),5,100,"UK"),
            ("2","12345","A",1,datetime(2011,1,11,12),10,100,"UK"),
            ("C3","12345","A",-1,datetime(2011,1,12,12),10,100,"UK"),
            ("4","12345","A",2,datetime(2011,1,20),10,100,"UK"),
            ("5","12345","A",1,datetime(2011,1,5),10,200,"UK"),
            ("6","12345","A",1,datetime(2011,1,21),10,300,"UK"),
            ("7","POST","POSTAGE",1,datetime(2011,1,5),100,100,"UK"),
            ("8","12345","A",1,datetime(2011,1,5),100,None,"UK")]
        self.con.executemany("INSERT OR IGNORE INTO lines(invoice,stock_code,description,quantity,invoice_date,unit_price,customer_id,country,source_sheet,row_hash,reason,is_return,value) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",[normalize(r) for r in rows+rows[:1]])
    def tearDown(self):self.con.close()
    def features(self):return extract_features(self.con,ROOT/"sql/customer_features.sql","2011-01-20")
    def test_frequency_counts_invoices_not_items(self):
        r=self.features().set_index("customer_id").loc["100"]
        self.assertEqual(r.frequency,2);self.assertEqual(r.purchase_lines,3)
        self.assertEqual(r.monetary,35);self.assertEqual(r.return_value,10)
        self.assertEqual(r.observed_net_value,25);self.assertEqual(r.unique_products,2)
    def test_window_gap_and_recency(self):
        r=self.features().set_index("customer_id").loc["100"]
        self.assertEqual(r.avg_gap_days,10);self.assertEqual(r.recency,9)
        self.assertEqual(r.avg_order_value,17.5)
    def test_exact_duplicate_removed(self):
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM lines").fetchone()[0],9)
    def test_cutoff_exclusive_and_future_customers_excluded(self):
        f=self.features();self.assertEqual(set(f.customer_id),{"100","200"})
        f["cluster"]=0;f["rfm_persona"]="test"
        h=holdout(self.con,f,"2011-01-20","2011-01-22").set_index("customer_id")
        self.assertEqual(h.loc["100","future_orders"],1)
        self.assertEqual(h.loc["100","future_gross_value"],20)
        self.assertEqual(h.loc["200","purchased_in_holdout"],0)
        self.assertEqual(len(h),2)
    def test_future_edits_do_not_change_features(self):
        a=self.features()
        self.con.execute("UPDATE lines SET value=999999 WHERE invoice_date >= '2011-01-20'")
        pd.testing.assert_frame_equal(a,self.features())
    def test_single_order_gap_is_missing(self):
        r=self.features().set_index("customer_id").loc["200"]
        self.assertTrue(pd.isna(r.avg_gap_days));self.assertEqual(r.single_order,1)
    def test_cleaning_reasons(self):
        reasons=dict(self.con.execute("SELECT invoice,reason FROM lines"))
        self.assertEqual(reasons["7"],"non_merchandise_code")
        self.assertEqual(reasons["8"],"missing_customer")
        self.assertEqual(reasons["C3"],"return")
    def test_censored_cohort_cells_not_zero(self):
        self.con.execute("UPDATE lines SET invoice_date='2011-02-10 00:00:00' WHERE customer_id='300'")
        c=cohort_retention(self.con,"2011-03-02")
        self.assertTrue(np.isnan(c.loc["2011-02",1]))
        self.assertEqual(c.loc["2011-01",1],0)
        self.assertEqual(c.loc["2011-02",0],1)

class TransformTests(unittest.TestCase):
    def test_transform_does_not_refit_on_future_data(self):
        x=pd.DataFrame({c:np.arange(1.,101.) for c in FEATURES})
        x["single_order"]=0;x["recent_order_share"]=.5
        prep=FeatureTransform().fit(x)
        before=prep.scaler_.mean_.copy()
        extreme=x.copy();extreme["monetary"]=1e12
        t=prep.transform(extreme)
        np.testing.assert_array_equal(prep.scaler_.mean_,before)
        self.assertTrue(np.isfinite(t).all())
    def test_missing_gap_finite_and_flag_kept(self):
        x=pd.DataFrame({c:[1.,2.,3.] for c in FEATURES});x["avg_gap_days"]=np.nan
        x["single_order"]=[0,0,1]
        p=FeatureTransform().fit(x)
        self.assertTrue(np.isfinite(p.transform(x)).all())
        self.assertEqual(p.high_["single_order"],1)
    def test_wilson_bounds(self):
        lo,hi=wilson(0,10);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
        lo,hi=wilson(10,10);self.assertAlmostEqual(hi,1);self.assertLess(lo,1)
    def test_persona_rule_priority(self):
        x=pd.DataFrame({"recency":[100,1,20,50],"frequency":[10,1,10,2],
                        "monetary":[1000,10,1000,20],"customer_age_days":[300,1,200,100]})
        p,_=add_personas(x)
        self.assertEqual(p.rfm_persona.iloc[0],"At-risk high-value")
        self.assertEqual(p.rfm_persona.iloc[1],"New / first purchase")
        self.assertEqual(p.rfm_persona.iloc[2],"High-value loyal")

if __name__=="__main__":unittest.main()
