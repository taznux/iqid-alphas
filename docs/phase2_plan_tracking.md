# Phase 2 Plan Tracking: Pipeline Integration & CLI Development

**Date**: June 19, 2025  
**Status**: Planning Phase  
**Previous Phase**: ✅ Phase 1 Complete (Modular core system)  
**Current Objective**: Integrate modular components into production-ready pipelines with CLI interfaces

---

## **Phase 2 Overview**

### **Objective**: 
Build production-ready pipelines that leverage the modular core components developed in Phase 1, creating a unified CLI interface and comprehensive pipeline orchestration system.

### **Architectural Approach**:
- **Pipeline-First Design**: Create three distinct pipelines (segmentation, alignment, coregistration)
- **Thin Interface Pattern**: Lightweight CLI scripts in `pipelines/` that delegate to `iqid_alphas/pipelines/`
- **Configuration-Driven**: JSON-based configuration system for flexible parameter management
- **Unified CLI**: Single entry point with subcommands for different pipelines
- **Modular Architecture**: No single module exceeds 500 lines; complex functionality split into focused packages

---

## **Phase 2 Success Criteria**

### ✅ **Technical Requirements**
- [ ] Three fully functional pipeline implementations
- [ ] Unified CLI interface with proper argument parsing
- [ ] Configuration management system
- [ ] Comprehensive logging and progress tracking
- [ ] Error handling and recovery mechanisms
- [ ] Performance optimization and parallel processing

### ✅ **Quality Requirements** 
- [ ] All pipelines pass validation tests
- [ ] CLI usability testing completed
- [ ] Performance benchmarks established
- [ ] Documentation complete and reviewed
- [ ] Integration tests with real data

### ✅ **Production Readiness**
- [ ] Batch processing capabilities
- [ ] Resource monitoring and optimization
- [ ] Comprehensive error reporting
- [ ] User-friendly help and documentation
- [ ] Automated testing and CI/CD integration

---

## **Phase 2 Task Breakdown**

### **Task 2.1: Pipeline Infrastructure Development** 

**Objective**: Create the foundational pipeline framework that orchestrates core modules.

**Sub-Task 2.1.1: Create Pipeline Base Classes**
- **Action**: Implement `iqid_alphas/pipelines/base.py` with abstract pipeline interface
- **Deliverable**: `BasePipeline` class with common functionality (logging, config, progress tracking)
- **Dependencies**: Phase 1 core modules
- **Estimated Effort**: 2-3 days

**Sub-Task 2.1.2: Implement Configuration System**
- **Action**: Create `iqid_alphas/configs/` directory with JSON configuration templates
- **Deliverable**: Configuration loader and validator classes
- **Files**: 
  - `iqid_alphas/configs/base_config.json`
  - `iqid_alphas/configs/segmentation_config.json`
  - `iqid_alphas/configs/alignment_config.json`
  - `iqid_alphas/configs/coregistration_config.json`
- **Estimated Effort**: 1-2 days

**Sub-Task 2.1.3: Create Progress Tracking System**
- **Action**: Implement progress reporting with logging integration
- **Deliverable**: `ProgressTracker` class with console and file output
- **Features**: Real-time progress bars, ETA calculation, performance metrics
- **Estimated Effort**: 1 day

---

### **Task 2.2: Segmentation Pipeline Implementation**

**Objective**: Create the Raw → Segmented validation pipeline using Phase 1 tissue segmentation components.

**Sub-Task 2.2.1: Core Pipeline Implementation**
- **Action**: Create `iqid_alphas/pipelines/segmentation.py`
- **Deliverable**: `SegmentationPipeline` class
- **Key Features**:
  - Integration with `TissueSeparator` from Phase 1
  - Batch processing of raw image directories
  - Ground truth validation using reverse mapping framework
  - Quality metric calculation (IoU, Dice, precision, recall)
- **Dependencies**: Phase 1 tissue segmentation, validation, batch processing modules
- **Estimated Effort**: 3-4 days

**Sub-Task 2.2.2: CLI Interface Creation**
- **Action**: Create `pipelines/segmentation.py` thin interface script
- **Deliverable**: Command-line tool with proper argument parsing
- **Features**:
  - Input/output directory specification
  - Configuration file support
  - Verbose logging options
  - Progress reporting
- **Estimated Effort**: 1 day

**Sub-Task 2.2.3: Validation and Testing**
- **Action**: Create comprehensive tests for segmentation pipeline
- **Deliverable**: `iqid_alphas/tests/pipelines/test_segmentation.py`
- **Coverage**: End-to-end pipeline testing, error handling, performance validation
- **Estimated Effort**: 1-2 days

---

### **Task 2.3: Alignment Pipeline Implementation**

**Objective**: Create the Segmented → Aligned validation pipeline using Phase 1 alignment components.

**Sub-Task 2.3.1: Core Pipeline Implementation**
- **Action**: Create `iqid_alphas/pipelines/alignment.py`
- **Deliverable**: `AlignmentPipeline` class
- **Key Features**:
  - Integration with modular alignment package from Phase 1
  - Bidirectional alignment processing
  - Ground truth validation using reverse mapping
  - Advanced quality metrics (SSIM, mutual information)
- **Dependencies**: Phase 1 alignment package, validation system
- **Estimated Effort**: 3-4 days

**Sub-Task 2.3.2: CLI Interface Creation**  
- **Action**: Create `pipelines/alignment.py` thin interface script
- **Deliverable**: Command-line tool for alignment validation
- **Features**: Similar CLI structure to segmentation with alignment-specific options
- **Estimated Effort**: 1 day

**Sub-Task 2.3.3: Advanced Alignment Methods**
- **Action**: Integrate bidirectional and stack alignment methods
- **Deliverable**: Enhanced alignment capabilities
- **Features**: Reference slice detection, propagation algorithms, noise robustness
- **Dependencies**: Phase 1 stack alignment components
- **Estimated Effort**: 2-3 days

---

### **Task 2.4: Coregistration Pipeline Implementation**

**Objective**: Create the iQID+H&E co-registration pipeline using Phase 1 coregistration components.

**Sub-Task 2.4.1: Core Pipeline Implementation**
- **Action**: Create `iqid_alphas/pipelines/coregistration.py`
- **Deliverable**: `CoregistrationPipeline` class  
- **Key Features**:
  - Integration with `MultiModalAligner` from Phase 1
  - Automated sample pairing and discovery
  - Multi-modal registration with mutual information
  - Proxy quality assessment (no ground truth available)
- **Dependencies**: Phase 1 coregistration package, batch processing
- **Estimated Effort**: 3-4 days

**Sub-Task 2.4.2: CLI Interface Creation**
- **Action**: Create `pipelines/coregistration.py` thin interface script  
- **Deliverable**: Command-line tool for coregistration
- **Features**: Multi-modal specific CLI options and validation
- **Estimated Effort**: 1 day

**Sub-Task 2.4.3: Quality Assessment Framework**
- **Action**: Implement proxy metrics for registration quality
- **Deliverable**: Quality assessment without ground truth
- **Features**: Confidence scoring, feature matching quality, registration consistency
- **Estimated Effort**: 2 days

---

### **Task 2.5: Unified CLI System**

**Objective**: Create a single entry point CLI that orchestrates all pipelines.

**Sub-Task 2.5.1: Main CLI Implementation**
- **Action**: Create `iqid_alphas/cli.py` and update `iqid_alphas/__main__.py`
- **Deliverable**: Unified CLI with subcommands
- **Features**:
  - `python -m iqid_alphas segmentation <args>`
  - `python -m iqid_alphas alignment <args>`  
  - `python -m iqid_alphas coregistration <args>`
  - Common options: --config, --verbose, --output-dir
- **Estimated Effort**: 2-3 days

**Sub-Task 2.5.2: Help System and Documentation**
- **Action**: Implement comprehensive help system
- **Deliverable**: Rich help text, examples, and error messages
- **Features**: Context-sensitive help, example commands, troubleshooting guides
- **Estimated Effort**: 1-2 days

**Sub-Task 2.5.3: Configuration Validation**
- **Action**: Add configuration file validation and error reporting
- **Deliverable**: Robust config handling with clear error messages
- **Features**: Schema validation, default value handling, configuration migration
- **Estimated Effort**: 1 day

---

### **Task 2.6: Performance & Production Optimization**

**Objective**: Optimize pipelines for production use with large datasets.

**Sub-Task 2.6.1: Parallel Processing Implementation**
- **Action**: Add parallel processing capabilities to all pipelines
- **Deliverable**: Multi-threaded/multi-process execution
- **Features**: Configurable worker count, memory management, progress aggregation
- **Dependencies**: Phase 1 batch processing framework
- **Estimated Effort**: 2-3 days

**Sub-Task 2.6.2: Memory Optimization**
- **Action**: Implement memory-efficient processing for large datasets
- **Deliverable**: Streaming data processing, garbage collection optimization
- **Features**: Lazy loading, memory monitoring, automatic cleanup
- **Estimated Effort**: 2 days

**Sub-Task 2.6.3: Logging and Monitoring**
- **Action**: Implement comprehensive logging and monitoring system
- **Deliverable**: Structured logging with performance metrics
- **Features**: JSON logs, performance profiling, resource usage tracking
- **Estimated Effort**: 1-2 days

---

### **Task 2.7: Integration Testing & Validation**

**Objective**: Ensure all components work together correctly with real data.

**Sub-Task 2.7.1: End-to-End Integration Tests**
- **Action**: Create comprehensive integration test suite
- **Deliverable**: `iqid_alphas/tests/integration/` test modules
- **Coverage**: Full pipeline execution, CLI testing, configuration validation
- **Estimated Effort**: 2-3 days

**Sub-Task 2.7.2: Real Data Validation**
- **Action**: Validate pipelines against known datasets
- **Deliverable**: Validation reports and benchmark results
- **Features**: Performance benchmarks, accuracy validation, regression testing
- **Estimated Effort**: 2-3 days

**Sub-Task 2.7.3: User Acceptance Testing**
- **Action**: Conduct usability testing of CLI interfaces
- **Deliverable**: User feedback incorporation and improvements
- **Features**: CLI usability, documentation clarity, error message quality
- **Estimated Effort**: 1-2 days

---

## **Phase 2 Architecture Overview**

### **Directory Structure**
```
iqid_alphas/
├── cli.py                    # Main CLI entry point
├── __main__.py              # Module execution entry
├── pipelines/               # Pipeline implementations
│   ├── base.py             # Base pipeline classes
│   ├── segmentation.py     # Segmentation pipeline
│   ├── alignment.py        # Alignment pipeline
│   └── coregistration.py   # Coregistration pipeline
├── configs/                # Configuration templates
│   ├── base_config.json
│   ├── segmentation_config.json
│   ├── alignment_config.json
│   └── coregistration_config.json
├── core/                   # Phase 1 modules (existing)
│   ├── validation/
│   ├── batch_processing/
│   ├── segmentation/
│   ├── alignment/
│   └── coregistration/
└── tests/                  # Comprehensive test suite
    ├── pipelines/          # Pipeline tests
    └── integration/        # Integration tests

pipelines/                  # Thin CLI interfaces
├── segmentation.py         # CLI wrapper for segmentation
├── alignment.py           # CLI wrapper for alignment
└── coregistration.py      # CLI wrapper for coregistration
```

### **Data Flow Architecture**
```
User Input → CLI Parser → Configuration → Pipeline → Core Modules → Output
     ↓           ↓            ↓            ↓           ↓           ↓
  Commands   Arguments    JSON Config   Orchestration  Processing   Results
```

---

## **Phase 2 Progress Tracker**

| Task ID | Module / Component | Key Deliverables | Status | Test Status | Estimated Effort |
|---------|-------------------|------------------|---------|-------------|------------------|
| **2.1** | Pipeline Infrastructure | Base classes, config system, progress tracking | ✅ **COMPLETED** | Tests Needed | 4-6 days |
| **2.2** | Segmentation Pipeline | Core pipeline + CLI + tests | ✅ **COMPLETED** | Tests Needed | 5-7 days |
| **2.3** | Alignment Pipeline | Core pipeline + CLI + advanced methods | � **IN PROGRESS** | Tests Needed | 6-8 days |
| **2.4** | Coregistration Pipeline | Core pipeline + CLI + quality assessment | � **IN PROGRESS** | Tests Needed | 6-7 days |
| **2.5** | Unified CLI System | Main CLI + help + config validation | � **IN PROGRESS** | Tests Needed | 4-6 days |
| **2.6** | Performance Optimization | Parallel processing + memory + logging | 🔄 **Not Started** | Not Started | 5-7 days |
| **2.7** | Integration Testing | End-to-end tests + real data validation | 🔄 **Not Started** | Not Started | 5-8 days |

**Total Estimated Effort**: 35-49 days (7-10 weeks)

---

## **Git Workflow Strategy**

### **Branch Strategy**
- **Main Branch**: `main` - Production-ready code only
- **Development Branch**: `develop` - Integration branch for completed features  
- **Feature Branches**: `feature/phase2-<component>` - Individual feature development
- **Current Working Branch**: `feature/phase2-pipeline-integration`

### **Commit Guidelines**
- **Atomic Commits**: Each commit represents one logical change
- **Descriptive Messages**: Clear, concise commit messages following conventional format
- **Frequent Commits**: Commit after each sub-task completion
- **Format**: `<type>(scope): <description>`
  - Types: feat, fix, test, docs, refactor, style, chore
  - Scopes: pipeline, cli, config, core, test

### **Push Strategy**
- **Daily Pushes**: Push work to feature branch at least once per day
- **Feature Completion**: Create PR when entire task is complete
- **Code Review**: All code reviewed before merging to develop
- **Integration**: Merge to develop after feature completion and review

### **Example Workflow**
```bash
# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/phase2-segmentation-pipeline

# Work and commit
git add -A
git commit -m "feat(pipeline): implement SegmentationPipeline base structure"

# Push daily
git push origin feature/phase2-segmentation-pipeline

# When feature complete, create PR to develop
# After review and approval, merge to develop
```

---

## **Dependencies and Prerequisites**

### **Phase 1 Dependencies (Completed)**
- ✅ Modular validation package
- ✅ Modular batch processing package  
- ✅ Tissue segmentation components
- ✅ Alignment package (SimpleAligner, EnhancedAligner, etc.)
- ✅ Coregistration package (MultiModalAligner)
- ✅ Reverse mapping framework
- ✅ Core utilities (io_utils, math)

### **External Dependencies**
- Python 3.11+ 
- Required packages: numpy, opencv-python, scikit-image, pytest
- Development tools: pre-commit, black, flake8

### **Data Dependencies**
- UCSF ReUpload dataset (for validation)
- DataPush1 dataset (for coregistration)
- Ground truth data (for segmentation and alignment validation)

---

## **Risk Assessment & Mitigation**

### **Technical Risks**

**Risk 1: Performance Degradation**
- **Impact**: New pipeline slower than current scripts
- **Mitigation**: Benchmark at each stage, optimize critical paths, parallel processing
- **Monitoring**: Performance tests with each feature completion

**Risk 2: Integration Complexity**  
- **Impact**: Core modules don't integrate smoothly in pipelines
- **Mitigation**: Comprehensive integration testing, incremental development
- **Monitoring**: Daily integration tests during development

**Risk 3: CLI Usability Issues**
- **Impact**: Difficult to use CLI reduces adoption
- **Mitigation**: User testing, clear documentation, intuitive interface design
- **Monitoring**: Regular usability testing sessions

### **Project Risks**

**Risk 1: Scope Creep**
- **Impact**: Additional features delay core functionality
- **Mitigation**: Strict scope definition, MVP approach, feature prioritization
- **Monitoring**: Regular scope reviews

**Risk 2: Timeline Pressure**
- **Impact**: Rushed implementation leads to technical debt
- **Mitigation**: Realistic estimates, parallel development, early feedback
- **Monitoring**: Weekly progress reviews

---

## **Success Metrics**

### **Functional Metrics**
- [ ] All three pipelines execute without errors on test data
- [ ] CLI interfaces provide comprehensive help and error handling
- [ ] Configuration system supports all required parameters
- [ ] Performance meets or exceeds current script performance

### **Quality Metrics**  
- [ ] 90%+ test coverage for all pipeline code
- [ ] All integration tests pass consistently
- [ ] User acceptance testing shows positive feedback
- [ ] Documentation completeness score > 95%

### **Production Readiness Metrics**
- [ ] Memory usage optimized for large datasets
- [ ] Parallel processing scales appropriately  
- [ ] Error handling covers all edge cases
- [ ] Logging provides actionable debugging information

---

## **Phase 2 Completion Criteria**

### **Required Deliverables**
1. **Three Production Pipelines**: Segmentation, Alignment, Coregistration
2. **Unified CLI System**: Single entry point with subcommands
3. **Configuration Management**: JSON-based parameter control
4. **Comprehensive Testing**: Unit, integration, and performance tests
5. **Performance Optimization**: Parallel processing and memory efficiency
6. **Documentation**: User guides, API docs, troubleshooting guides

### **Quality Gates**
- [ ] All automated tests pass (unit + integration)
- [ ] Performance benchmarks meet targets
- [ ] Code review completed and approved
- [ ] User acceptance testing successful
- [ ] Documentation review completed

### **Ready for Phase 3 Criteria**
- [ ] All Phase 2 deliverables complete and tested
- [ ] Pipelines validated against real datasets
- [ ] Production deployment documentation ready
- [ ] Team training on new system completed

---

**Next Phase**: Phase 3 - Production Deployment & Advanced Features
- Production environment setup
- Advanced analytics and reporting
- API development for remote processing
- Cloud integration and scalability
