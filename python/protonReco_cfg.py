import FWCore.ParameterSet.Config as cms

def setupProtonReco(process,xangle):

    # random seeds
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                       sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
                                                       generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
                                                       SmearingGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849)),
                                                       beamDivergenceVtxGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849))  
                                                       )


    # geometry
    process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")
    del(process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles[-1])
    process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("Validation/CTPPS/test_2017/RP_Dist_Beam_Cent.xml")
    
    # fast simulation
    process.load('SimCTPPS.OpticsParameterisation.year_2017_OF.ctppsFastProtonSimulation_cfi')
    process.ctppsFastProtonSimulation.hepMCTag = cms.InputTag('beamDivergenceVtxGenerator')

    appertures={120:{45:0.143,56:0.17},
                130:{45:0.148,56:0.18},
                140:{45:0.153,56:0.19},
                150:{45:0.158,56:0.20}}
    process.ctppsFastProtonSimulation.xangle = xangle
    process.ctppsFastProtonSimulation.empiricalAperture45_xi0 = appertures[xangle][45]
    process.ctppsFastProtonSimulation.empiricalAperture56_xi0 = appertures[xangle][56]

    process.ctppsFastProtonSimulation.produceScoringPlaneHits = False
    process.ctppsFastProtonSimulation.produceRecHits = True
    process.ctppsFastProtonSimulation.checkApertures = False
    process.ctppsFastProtonSimulation.useEmpiricalApertures = True 
    process.ctppsFastProtonSimulation.produceHitsRelativeToBeam = True 
    process.ctppsFastProtonSimulation.roundToPitch = True

    # beam-smearing settings
    process.load("IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi")
    process.beamDivergenceVtxGenerator.src = cms.InputTag('generatorSmeared')
    process.beamDivergenceVtxGenerator.simulateBeamDivergence = True
    process.beamDivergenceVtxGenerator.simulateVertex = True
    
    # values in rad
    process.beamDivergenceVtxGenerator.beamDivergenceX = 30E-6
    process.beamDivergenceVtxGenerator.beamDivergenceY = 30E-6

    # values in cm
    process.beamDivergenceVtxGenerator.vertexMeanX = 0.02476
    process.beamDivergenceVtxGenerator.vertexMeanY = -0.06923
    process.beamDivergenceVtxGenerator.vertexMeanZ = -0.8342

    process.beamDivergenceVtxGenerator.vertexSigmaX = 0.
    process.beamDivergenceVtxGenerator.vertexSigmaY = 0.
    process.beamDivergenceVtxGenerator.vertexSigmaZ = 0.
    
    # local track reco
    process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
    process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulation')

    process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
    
    process.load("RecoCTPPS.PixelLocal.ctppsPixelLocalTracks_cfi")
    process.ctppsPixelLocalTracks.label = "ctppsFastProtonSimulation"
    
    process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff')
    process.ctppsLocalTrackLiteProducer.includeDiamonds = False


    process.load("RecoCTPPS.ProtonReconstruction.year_2017_OF.ctppsProtonReconstructionOF_cfi")
    process.ctppsProtonReconstructionOFDB.applyExperimentalAlignment = False # do not use alignment for LHC data
    
    process.ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
                                                beamEnergy = cms.double(6500),
                                                xangle = cms.double(xangle)
                                                )
    
    process.pps_reco_step = cms.Path(
        process.ctppsProtonReconstructionOFDB
        )

    process.pps_fastsim = cms.Path(
        process.beamDivergenceVtxGenerator*
        process.ctppsFastProtonSimulation
        )

    process.pps_simulation_step = cms.Path(
        process.totemRPUVPatternFinder
        * process.totemRPLocalTrackFitter
        * process.ctppsPixelLocalTracks
        * process.ctppsLocalTrackLiteProducer
        )
