#include "TopLJets2015/TopAnalysis/interface/KinematicsTools.h"
#include <iostream>


//
Float_t computeRecoilMT(TLorentzVector &a, TVector2 &h){
  float mt2_val = pow(a.Pt(),2);
  mt2_val += a.Pt()*((a.Vect().XYvector()+h).Mod());
  mt2_val += a.Vect().XYvector()*h;
  mt2_val *= 2;
  return TMath::Sqrt(mt2_val);
}

//
Float_t computeMT(TLorentzVector &a, TLorentzVector &b)
{
  return TMath::Sqrt(2*a.Pt()*b.Pt()*(1-TMath::Cos(a.DeltaPhi(b))));
}

Float_t computeMR(TLorentzVector &hem1, TLorentzVector &hem2){
  return sqrt(pow(hem1.P() + hem2.P(), 2) - pow(hem1.Pz() + hem2.Pz(), 2));
}

Float_t computeRsq(TLorentzVector &hem1, TLorentzVector &hem2, TLorentzVector &pfMet){
  Float_t mR = computeMR(hem1, hem2);
  Float_t term1 = pfMet.Pt()/2*(hem1.Pt() + hem2.Pt());
  Float_t term2 = pfMet.Px()/2*(hem1.Px() + hem2.Px()) + pfMet.Py()/2*(hem1.Py() + hem2.Py()); //dot product of MET with (p1T + p2T)
  Float_t mTR = sqrt(term1 - term2);
  return (mTR / mR) * (mTR / mR);
}

Float_t computePhiStar(TLorentzVector &lm,TLorentzVector &lp){
  Float_t dphill        = fabs(lm.DeltaPhi(lp));
  Float_t phi_acop      = TMath::Pi()-dphill;
  Float_t costhetastar  = TMath::TanH(0.5*(lm.Eta()-lp.Eta()));
  Float_t sin2thetastar = costhetastar>1 ? 0.0 : (1.0 - TMath::Sqrt(costhetastar));
  Float_t phistar       = TMath::Tan(0.5*phi_acop) * TMath::Sqrt( sin2thetastar );
  return phistar;
}

Float_t computeCosThetaStar(TLorentzVector &lm,TLorentzVector &lp){
  TLorentzVector dil(lm+lp);
  Float_t costhetaCS((dil.Pz()>0 ? 1.0 : -1.0)/dil.M());
  costhetaCS *= (lm.E() + lm.Pz()) * (lp.E() - lp.Pz()) - (lm.E() - lm.Pz()) * (lp.E() + lp.Pz());
  costhetaCS /= TMath::Sqrt( pow(dil.M(),2) + pow(dil.Pt(),2) );
  return costhetaCS;
}

Float_t computeAcoplanarity(TLorentzVector &lm,TLorentzVector &lp){
  return 1.0-fabs(lm.DeltaPhi(lp)/TMath::Pi());
}

//
ValueCollection_t calcCsi(TLorentzVector &a, TLorentzVector &b, Float_t sqrts){
  
  Float_t csip(a.Pt()*exp(a.Eta())+b.Pt()*exp(b.Eta()));
  Float_t csim(a.Pt()*exp(-a.Eta())+b.Pt()*exp(-b.Eta()));
  ValueCollection_t toRet;
  toRet.push_back( Value_t(csip/sqrts,0.) );
  toRet.push_back( Value_t(csim/sqrts,0.) );
  return toRet;
}
