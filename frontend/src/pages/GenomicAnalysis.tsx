import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  LinearProgress,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Snackbar,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Science,
  ExpandMore,
  Info,
  Warning,
  Error,
  CheckCircle,
  Timeline,
  Assessment,
  Biotech,
  Psychology,
  Medication,
  Favorite,
  LocalHospital,
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

// Types
interface GenomicVariant {
  chromosome: string;
  position: number;
  reference: string;
  alternate: string;
  quality: number;
  depth: number;
  genotype: string;
  gene?: string;
  impact?: 'high' | 'moderate' | 'low' | 'modifier';
  consequence?: string;
  clinvar?: string;
  population_frequency?: number;
}

interface PharmacogenomicProfile {
  gene: string;
  genotype: string;
  phenotype: 'poor_metabolizer' | 'intermediate_metabolizer' | 'normal_metabolizer' | 'ultrarapid_metabolizer';
  drugs: string[];
  recommendations: string[];
}

interface DiseaseRisk {
  disease: string;
  risk_score: number;
  confidence: number;
  contributing_variants: GenomicVariant[];
  preventive_measures: string[];
}

interface AnalysisResult {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  variants: GenomicVariant[];
  pharmacogenomics: PharmacogenomicProfile[];
  disease_risks: DiseaseRisk[];
  ancestry: any;
  recommendations: string[];
  warnings: string[];
  created_at: string;
  completed_at?: string;
}

const GenomicAnalysis: React.FC = () => {
  const { user } = useAuth();

  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'warning' | 'info',
  });

  // Load analysis history
  useEffect(() => {
    loadAnalysisHistory();
  }, [user]);

  const loadAnalysisHistory = async () => {
    setLoading(true);
    try {
      // Mock data - replace with actual API call
      const mockResults: AnalysisResult[] = [
        {
          job_id: 'analysis_001',
          status: 'completed',
          progress: 100,
          variants: [
            {
              chromosome: '7',
              position: 117199646,
              reference: 'G',
              alternate: 'A',
              quality: 45.0,
              depth: 30,
              genotype: 'A/A',
              gene: 'CFTR',
              impact: 'moderate',
              consequence: 'missense_variant',
              clinvar: 'Pathogenic',
              population_frequency: 0.001,
            },
            {
              chromosome: '12',
              position: 21334919,
              reference: 'C',
              alternate: 'T',
              quality: 52.0,
              depth: 35,
              genotype: 'C/T',
              gene: ' PAH',
              impact: 'high',
              consequence: 'splice_donor_variant',
              population_frequency: 0.0001,
            },
          ],
          pharmacogenomics: [
            {
              gene: 'CYP2D6',
              genotype: 'CYP2D6*1/CYP2D6*4',
              phenotype: 'intermediate_metabolizer',
              drugs: ['codeine', 'tamoxifen', 'paroxetine'],
              recommendations: [
                'Monitor for reduced drug efficacy',
                'Consider alternative medications',
                'Consult pharmacist for dosage adjustments',
              ],
            },
            {
              gene: 'CYP2C19',
              genotype: 'CYP2C19*1/CYP2C19*2',
              phenotype: 'intermediate_metabolizer',
              drugs: ['clopidogrel', 'omeprazole', 'citalopram'],
              recommendations: [
                'Consider genetic testing before prescribing',
                'Monitor therapeutic drug levels',
                'Adjust dosage based on genotype',
              ],
            },
          ],
          disease_risks: [
            {
              disease: 'Cardiovascular Disease',
              risk_score: 0.15,
              confidence: 0.85,
              contributing_variants: [],
              preventive_measures: [
                'Regular cardiovascular screening',
                'Lifestyle modifications',
                'Blood pressure monitoring',
              ],
            },
            {
              disease: 'Type 2 Diabetes',
              risk_score: 0.22,
              confidence: 0.78,
              contributing_variants: [],
              preventive_measures: [
                'Weight management',
                'Regular exercise',
                'Blood glucose monitoring',
              ],
            },
          ],
          ancestry: {
            primary_ancestry: 'European',
            admixture: {
              European: 0.85,
              East_Asian: 0.10,
              African: 0.05,
            },
          },
          recommendations: [
            'Schedule consultation with genetic counselor',
            'Consider preventive screening for identified risks',
            'Review medication regimens with pharmacogenomic profile',
            'Update family medical history',
          ],
          warnings: [
            'Several variants of uncertain significance identified',
            'Limited clinical evidence for some risk associations',
          ],
          created_at: '2024-01-10T09:00:00Z',
          completed_at: '2024-01-10T14:30:00Z',
        },
      ];

      setAnalysisResults(mockResults);
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to load analysis history',
        severity: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setSnackbar({
        open: true,
        message: 'Please select a file to upload',
        severity: 'warning',
      });
      return;
    }

    setUploading(true);
    try {
      // Mock upload - replace with actual API call
      const formData = new FormData();
      formData.append('genome_file', selectedFile);
      formData.append('patient_id', user?.id || '');
      formData.append('analysis_type', 'comprehensive');

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));

      setSnackbar({
        open: true,
        message: 'Genome uploaded successfully. Analysis will begin shortly.',
        severity: 'success',
      });

      setSelectedFile(null);
      // Refresh analysis history
      loadAnalysisHistory();
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to upload genome file',
        severity: 'error',
      });
    } finally {
      setUploading(false);
    }
  };

  const handleViewDetails = (result: AnalysisResult) => {
    setSelectedResult(result);
    setDetailDialogOpen(true);
  };

  const getStatusColor = (status: AnalysisResult['status']) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'info';
      case 'queued': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getVariantImpactColor = (impact?: string) => {
    switch (impact) {
      case 'high': return 'error';
      case 'moderate': return 'warning';
      case 'low': return 'info';
      case 'modifier': return 'default';
      default: return 'default';
    }
  };

  const getPhenotypeColor = (phenotype: string) => {
    switch (phenotype) {
      case 'poor_metabolizer': return 'error';
      case 'intermediate_metabolizer': return 'warning';
      case 'normal_metabolizer': return 'success';
      case 'ultrarapid_metabolizer': return 'info';
      default: return 'default';
    }
  };

  const formatRiskScore = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Genomic Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Comprehensive genetic analysis and personalized medicine insights
        </Typography>
      </Box>

      {/* Upload Section */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload Genome Data
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Upload your genomic data file (VCF, BAM, FASTQ formats supported)
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button variant="outlined" component="label">
              Choose File
              <input
                type="file"
                hidden
                accept=".vcf,.bam,.fastq,.fq,.gz"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
            </Button>

            {selectedFile && (
              <Typography variant="body2">
                Selected: {selectedFile.name}
              </Typography>
            )}

            <Button
              variant="contained"
              onClick={handleFileUpload}
              disabled={!selectedFile || uploading}
              startIcon={<Science />}
            >
              {uploading ? 'Uploading...' : 'Start Analysis'}
            </Button>
          </Box>

          {uploading && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Uploading and processing genome data...
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Analysis History */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Analysis History
          </Typography>

          {loading ? (
            <LinearProgress sx={{ mt: 2 }} />
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Job ID</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Completed</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {analysisResults.map((result) => (
                    <TableRow key={result.job_id}>
                      <TableCell>{result.job_id}</TableCell>
                      <TableCell>
                        <Chip
                          label={result.status}
                          color={getStatusColor(result.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={result.progress}
                            sx={{ width: 100 }}
                          />
                          <Typography variant="body2">
                            {result.progress}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {new Date(result.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        {result.completed_at
                          ? new Date(result.completed_at).toLocaleDateString()
                          : '-'
                        }
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleViewDetails(result)}
                          disabled={result.status !== 'completed'}
                        >
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Detailed Results Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: { height: '80vh' }
        }}
      >
        <DialogTitle>
          Genomic Analysis Results - {selectedResult?.job_id}
        </DialogTitle>
        <DialogContent sx={{ overflow: 'auto' }}>
          {selectedResult && (
            <Box>
              {/* Variants */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">
                    <Biotech sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Genetic Variants ({selectedResult.variants.length})
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Gene</TableCell>
                          <TableCell>Variant</TableCell>
                          <TableCell>Impact</TableCell>
                          <TableCell>Genotype</TableCell>
                          <TableCell>ClinVar</TableCell>
                          <TableCell>Frequency</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedResult.variants.map((variant, index) => (
                          <TableRow key={index}>
                            <TableCell>{variant.gene || 'N/A'}</TableCell>
                            <TableCell>
                              {variant.chromosome}:{variant.position} {variant.reference}>{variant.alternate}
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={variant.impact || 'unknown'}
                                color={getVariantImpactColor(variant.impact)}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{variant.genotype}</TableCell>
                            <TableCell>{variant.clinvar || 'N/A'}</TableCell>
                            <TableCell>
                              {variant.population_frequency
                                ? `${(variant.population_frequency * 100).toFixed(3)}%`
                                : 'N/A'
                              }
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>

              {/* Pharmacogenomics */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">
                    <Medication sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Pharmacogenomics
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {selectedResult.pharmacogenomics.map((profile, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6">{profile.gene}</Typography>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Genotype: {profile.genotype}
                            </Typography>
                            <Chip
                              label={profile.phenotype.replace('_', ' ')}
                              color={getPhenotypeColor(profile.phenotype)}
                              sx={{ mb: 1 }}
                            />
                            <Typography variant="body2" sx={{ mb: 1 }}>
                              <strong>Drugs:</strong> {profile.drugs.join(', ')}
                            </Typography>
                            <List dense>
                              {profile.recommendations.map((rec, idx) => (
                                <ListItem key={idx}>
                                  <ListItemIcon>
                                    <Info fontSize="small" />
                                  </ListItemIcon>
                                  <ListItemText primary={rec} />
                                </ListItem>
                              ))}
                            </List>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Disease Risks */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">
                    <Favorite sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Disease Risk Assessment
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {selectedResult.disease_risks.map((risk, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6">{risk.disease}</Typography>
                            <Typography variant="h4" color="primary" sx={{ mb: 1 }}>
                              {formatRiskScore(risk.risk_score)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              Confidence: {(risk.confidence * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" sx={{ mb: 1 }}>
                              <strong>Preventive Measures:</strong>
                            </Typography>
                            <List dense>
                              {risk.preventive_measures.map((measure, idx) => (
                                <ListItem key={idx}>
                                  <ListItemText primary={measure} />
                                </ListItem>
                              ))}
                            </List>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Recommendations and Warnings */}
              {(selectedResult.recommendations.length > 0 || selectedResult.warnings.length > 0) && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">
                      <LocalHospital sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Recommendations & Warnings
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    {selectedResult.recommendations.map((rec, index) => (
                      <Alert key={index} severity="info" sx={{ mb: 1 }}>
                        {rec}
                      </Alert>
                    ))}
                    {selectedResult.warnings.map((warning, index) => (
                      <Alert key={index} severity="warning" sx={{ mb: 1 }}>
                        {warning}
                      </Alert>
                    ))}
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
      >
        <Alert
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default GenomicAnalysis;
