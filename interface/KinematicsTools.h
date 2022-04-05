#ifndef _kinematicstools_h_
#define _kinematicstools_h_

#include "TLorentzVector.h"
#include "TVector2.h"
#include "TopLJets2015/TopAnalysis/interface/MiniEvent.h"

typedef std::pair<Float_t,Float_t> Value_t;
typedef std::vector<Value_t> ValueCollection_t;

Float_t computeMT(TLorentzVector &a, TLorentzVector &b);
Float_t computeRecoilMT(TLorentzVector &a, TVector2 &h);
Float_t computePhiStar(TLorentzVector &lm,TLorentzVector &lp);
Float_t computeMR(TLorentzVector &a,TLorentzVector &b);
Float_t computeRsq(TLorentzVector &a,TLorentzVector &b,TLorentzVector &met);
Float_t computeCosThetaStar(TLorentzVector &lm,TLorentzVector &lp);
Float_t computeAcoplanarity(TLorentzVector &lm,TLorentzVector &lp);
Float_t computePhiStar(TLorentzVector &lm,TLorentzVector &lp);
ValueCollection_t calcCsi(TLorentzVector &a, TLorentzVector &b, Float_t sqrts=13000.);

#endif
