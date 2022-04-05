#ifndef _ExclusiveZX_h_
#define _ExclusiveZX_h_

#include "TLorentzVector.h"
#include "TopLJets2015/TopAnalysis/interface/ObjectTools.h"
#include "TopLJets2015/TopAnalysis/interface/SelectionTools.h"

void RunExclusiveZX(const TString filename,
                    TString outname,
                    Int_t channelSelection, 
                    Int_t chargeSelection, 
                    TH1F *normH, 
                    TH1F *genPU,
                    TString era,
                    Bool_t debug=false);

#endif
